#!/usr/bin/env python3
import sys
import argparse
import os
import csv
from warnings import catch_warnings
import torch
import logging
import speechbrain as sb
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import run_on_main
"""Recipe for training a Transformer ASR system with CommonVoice
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with (CTC/Att joint) beamsearch.

To run this recipe, do the following:
> python train.py hparams/transformer.yaml

With the default hyperparameters, the system employs a convolutional frontend (ContextNet) and a transformer.
The decoder is based on a Transformer decoder.

The neural network is trained on both CTC and negative-log likelihood
targets and sub-word units estimated with Byte Pairwise Encoding (BPE)
are used as basic recognition tokens.

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE),
training split (e.g, train-clean 100 rather than the full one), and many
other possible variations.


Authors
 * Titouan Parcollet 2021
 * Jianyuan Zhong 2020
"""

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.core.Brain):
    # def __init__(
    #     self,
    #     modules=None,
    #     opt_class=None,
    #     hparams=None,
    #     run_opts=None,
    #     checkpointer=None,
    # ):
    #     super().__init__(modules, opt_class, hparams, run_opts, checkpointer,)
    #     # self.tr_acc_csv_h = open(hparams['train_acc_stats_file'], 'a', newline='')
    #     # self.tr_loss_csv_h = open(hparams['train_loss_stats_file'], 'a', newline='')

    #     # self.tr_acc_wrtr = csv.writer(self.tr_acc_csv_h, delimiter=',',
    #     #                     quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     # self.tr_loss_wrtr = csv.writer(self.tr_loss_csv_h, delimiter=',',
    #     #                     quotechar='|', quoting=csv.QUOTE_MINIMAL)


    #     # self.tr_acc_wrtr.writerow([1, 4, 5])

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        
        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.hparams.normalize(feats, wav_lens, epoch=current_epoch)

        # Augmentation
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                feats = self.hparams.augmentation(feats)

        # forward modules
        src = self.modules.CNN(feats)
        enc_out, pred = self.modules.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index
        )

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # Compute outputs
        hyps = None
        if stage == sb.Stage.TRAIN:
            hyps = None
        elif stage == sb.Stage.VALID:
            hyps = None
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0:
                hyps, _ = self.hparams.beam_searcher(enc_out.detach(), wav_lens)
        elif stage == sb.Stage.TEST:
            hyps, _ = self.hparams.beam_searcher(enc_out.detach(), wav_lens)

        return p_ctc, p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        (p_ctc, p_seq, wav_lens, predicted_tokens,) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )
        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or (
                stage == sb.Stage.TEST
            ):
                # Decode token terms to words
                predicted_words = self.tokenizer(
                    predicted_tokens, task="decode_from_list"
                )

                # Convert indices to words
                target_words = undo_padding(tokens, tokens_lens)
                target_words = self.tokenizer(
                    target_words, task="decode_from_list"
                )
                self.wer_metric.append(ids, predicted_words, target_words)
                self.cer_metric.append(ids, predicted_words, target_words)

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""

        # check if we need to switch optimizer
        # if so change the optimizer from Adam to SGD
        self.check_and_reset_optimizer()

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        tokens_eos, tokens_eos_lens = batch.tokens_eos
        (_, p_seq, _, _,) = predictions
        self.acc_tr_metric.append(p_seq, tokens_eos, tokens_eos_lens)

        self.tr_acc_arr.append(self.acc_tr_metric.summarize())
        self.tr_loss_arr.append(loss.detach().cpu().numpy())

        # self.tr_stats["TR_ACC"][str(self.step)] = self.acc_tr_metric.summarize()
        # self.tr_stats["TR_LOSS"][str(self.step)] = loss.detach().cpu()

        # normalize the loss by gradient_accumulation step
        (loss / self.hparams.gradient_accumulation).backward()

        if self.step % self.hparams.gradient_accumulation == 0:
            # gradient clipping & early stop if loss is not fini
            self.check_gradients(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            # anneal lr every update
            self.hparams.noam_annealing(self.optimizer)


        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()
        else:
            self.tr_acc_csv_h = open(hparams['train_acc_stats_file'], 'a', newline='')
            self.tr_loss_csv_h = open(hparams['train_loss_stats_file'], 'a', newline='')

            self.tr_acc_wrtr = csv.writer(self.tr_acc_csv_h, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            self.tr_loss_wrtr = csv.writer(self.tr_loss_csv_h, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

            self.tr_acc_arr = [epoch]
            self.tr_loss_arr = [epoch]
            # self.tr_stats["TR_ACC"] = {}
            # self.tr_stats["TR_LOSS"] = {}

            # TODO: Check if needed
            self.acc_tr_metric = self.hparams.acc_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

            # self.tr_stats["TR_ACC"]['epoch']

            self.tr_acc_wrtr.writerow(self.tr_acc_arr)
            self.tr_loss_wrtr.writerow(self.tr_loss_arr)

            self.tr_acc_csv_h.close()
            self.tr_loss_csv_h.close()

            # torch.save(self.tr_stats["TR_ACC"][epoch], '{}.pt'.format(
            #     self.hparams.train_acc_stats_file))
            # torch.save(self.tr_stats["TR_ACC"], '{}_ep{}.pt'.format(
                # self.hparams.train_acc_stats_file, epoch))

            # torch.save(self.tr_stats["TR_LOSS"], '{}_ep{}.pt'.format(
            #     self.hparams.train_loss_stats_file, epoch))

            # self.train_stats['comp_tr_epoch'] = epoch
            # torch.save(self.tr_stats, self.hparams.train_stats_file)
            # stage_stats["ACC"][str(epoch)] = self.acc_tr_metric.summarize()
            # plt.figure(figsize=(8, 4), dpi=100)
            # plt.plot(stage_loss)
            # plt.grid()
            # plt.show()
            # plt.savefig

        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")
                stage_stats["CER"] = self.cer_metric.summarize("error_rate")

                self.tr_stats['WER']['epoch'].append(epoch)
                self.tr_stats['WER']['wer'].append(stage_stats["WER"])

                self.tr_stats['CER']['epoch'].append(epoch)
                self.tr_stats['CER']['cer'].append(stage_stats["CER"])

                # self.tr_stats['CER'].append(stage_stats["CER"])
                torch.save(self.tr_stats, self.hparams.train_stats_file)

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():

            # report different epoch stages according current stage
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.stage_one_epochs:
                lr = self.hparams.noam_annealing.current_lr
                steps = self.hparams.noam_annealing.n_steps
                optimizer = self.optimizer.__class__.__name__
            else:
                lr = self.hparams.lr_sgd
                steps = -1
                optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                # "optimizer": optimizer,
            }

            self.tr_stats['epoch'].append(epoch)
            self.tr_stats['tr_loss'].append(self.train_stats['loss'])
            self.tr_stats['val_loss'].append(stage_stats['loss'])
            self.tr_stats['lr'].append(lr)
            self.tr_stats['optimizer'].append(optimizer)
            self.tr_stats['ACC'].append(stage_stats["ACC"])
            torch.save(self.tr_stats, self.hparams.train_stats_file)


            # Plot important graphs

            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
                verbose=True
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
            )

        elif stage == sb.Stage.TEST:

            self.tr_stats['tst_loss'].append(stage_stats['loss'])
            torch.save(self.tr_stats, self.hparams.train_stats_file)
            
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
                verbose=True
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

    def check_and_reset_optimizer(self):
        """reset the optimizer if training enters stage 2"""
        current_epoch = self.hparams.epoch_counter.current
        if not hasattr(self, "switched"):
            self.switched = False
            if isinstance(self.optimizer, torch.optim.SGD):
                self.switched = True

        if self.switched is True:
            return

        if current_epoch > self.hparams.stage_one_epochs:
            self.optimizer = self.hparams.SGD(self.modules.parameters())

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer", self.optimizer)

            self.switched = True

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp.

        Default implementation compiles the jit modules, initializes
        optimizers, and loads the latest checkpoint to resume training.
        """
        # Run this *after* starting all processes since jit modules cannot be
        # pickled.
        self._compile_jit()

        # Wrap modules with parallel backend after jit
        self._wrap_distributed()

        # Initialize optimizers after parameters are configured
        self.init_optimizers()

        # Load latest checkpoint to check to current epoch number
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                device=torch.device(self.device)
            )

            # try:
            #     self.tr_stats = torch.load(self.hparams.train_stats_file)
            # except:
            #     print('file {} not found'.format(self.hparams.train_stats_file))
                # self.tr_stats = {
                #     'epoch': [],
                #     'tr_loss': [],
                #     'val_loss': [],
                #     'tst_loss': [],
                #     'lr': [],
                #     'optimizer': [],
                #     'ACC': [],
                #     'TR_ACC': {},
                #     'TR_LOSS': {},
                #     'WER': {
                #         'epoch': [],
                #         'wer': []
                #     },
                #     'CER': {
                #         'epoch': [],
                #         'cer': []
                #     }
                # }
        # else:
        #     self.tr_stats = {
        #             'epoch': [],
        #             'tr_loss': [],
        #             'val_loss': [],
        #             'tst_loss': [],
        #             'lr': [],
        #             'optimizer': [],
        #             'ACC': [],
        #             'TR_ACC': {},
        #             'TR_LOSS': {},
        #             'WER': {
        #                 'epoch': [],
        #                 'wer': []
        #             },
        #             'CER': {
        #                 'epoch': [],
        #                 'cer': []
        #             }
        #     }


        # if the model is resumed from stage two, reinitialize the optimizer
        current_epoch = self.hparams.epoch_counter.current
        if current_epoch > self.hparams.stage_one_epochs:
            self.optimizer = self.hparams.SGD(self.modules.parameters())

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer", self.optimizer)

        # Load latest checkpoint to resume training if interrupted
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                device=torch.device(self.device)
            )


# Define custom data procedure
def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    # We also sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder},
    )

    # We also sort the validation data so it is faster to validate
    test_data = test_data.filtered_sorted(
        sort_key="duration",
        key_max_value={"duration": hparams["avoid_if_longer_than_test"]},
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        info = torchaudio.info(wav)
        sig, _ = sb.dataio.dataio.read_audio(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_data


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)


    if not os.path.isfile('{}/save/train.csv'.format(hparams['output_folder'])):
        os.system('cp -rv ref_dir/* {}/'.format(hparams['output_folder']))

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Dataset preparation (parsing CommonVoice)
    from common_voice_prepare import prepare_common_voice  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Due to DDP, we do the preparation ONLY on the main python process
    run_on_main(
        prepare_common_voice,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "train_tsv_file": hparams["train_tsv_file"],
            "dev_tsv_file": hparams["dev_tsv_file"],
            "test_tsv_file": hparams["test_tsv_file"],
            "accented_letters": hparams["accented_letters"],
            "language": hparams["language"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Defining tokenizer and loading it
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        annotation_train=hparams["train_csv"],
        annotation_read="wrd",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_data = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    asr_brain.tokenizer = tokenizer




    try:
        asr_brain.tr_stats = torch.load(hparams['train_stats_file'])
    except:
        print('file {} not found'.format(hparams['train_stats_file']))

        tr_num_samples = hparams['train_dataloader_opts'].get('num_samples', 759546)
        val_num_samples = hparams['valid_dataloader_opts'].get('num_samples', 16264)
        tr_param = {
            'max_epoch': hparams['number_of_epochs'],
            'tr_num_samples': tr_num_samples,
            'tr_batch': hparams['train_dataloader_opts']['batch_size'],
            'val_batch': hparams['valid_dataloader_opts']['batch_size'],
            'val_num_samples': val_num_samples,
        }
        asr_brain.tr_stats = {
            'epoch': [],
            'tr_loss': [],
            'val_loss': [],
            'tst_loss': [],
            'lr': [],
            'optimizer': [],
            'ACC': [],
            'TR_ACC': {},
            'TR_LOSS': {},
            'WER': {
                'epoch': [],
                'wer': []
            },
            'CER': {
                'epoch': [],
                'cer': []
            }
        }

        torch.save(tr_param, hparams['train_stats_param_file'])

    
    try:
        os.mkdir(hparams['train_stat_dir'])
    except:
        print('Training stat dir exists')


    if not run_opts['skip_train']:
        # Training
        asr_brain.fit(
            asr_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["train_dataloader_opts"],
            valid_loader_kwargs=hparams["valid_dataloader_opts"],
        )

    # Test
    asr_brain.hparams.wer_file = hparams["output_folder"] + "/wer_test.txt"
    asr_brain.evaluate(
        test_data,
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

    asr_brain.hparams.wer_file = hparams["output_folder"] + "/wer_valid.txt"
    asr_brain.evaluate(
        valid_data,
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
