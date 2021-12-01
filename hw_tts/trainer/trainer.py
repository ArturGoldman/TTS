import random
from random import shuffle

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import PIL
from torchvision.transforms import ToTensor

from hw_tts.base import BaseTrainer
from hw_tts.logger.utils import plot_spectrogram_to_buf
from hw_tts.utils import inf_loop, MetricTracker
from hw_tts.aligner import Batch, GraphemeAligner
from hw_tts.model import Vocoder


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion_fs,
            criterion_dp,
            optimizer,
            config,
            device,
            data_loader,
            val_data_loader=None,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True
    ):
        super().__init__(model, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.data_loader = data_loader
        self.val_data_loader = val_data_loader
        self.criterion_fs = criterion_fs
        self.criterion_dp = criterion_dp
        self.vocoder = Vocoder().to(self.device)

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.lr_scheduler = lr_scheduler
        self.log_step = 50
        self.galigner = GraphemeAligner().to(self.device)

        self.train_metrics = MetricTracker(
            "loss_fs", "loss_dp", "grad norm", writer=self.writer
        )

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.data_loader, desc="train", total=self.len_epoch)
        ):
            to_log = False
            if batch_idx + 1 == self.len_epoch or batch_idx + 1 == self.len_epoch // 2:
                to_log = True
            try:
                l_ft, l_dp = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                    to_log=to_log
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0 or (batch_idx + 1 == self.len_epoch and epoch == self.epochs):
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss_fs: {:.6f} Loss_dp: {:.6f}".format(
                        epoch, self._progress(batch_idx), l_ft, l_dp
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_scalars(self.train_metrics)
            if batch_idx + 1 >= self.len_epoch:
                break

        self._valid_example()

        log = self.train_metrics.result()

        return log

    def process_batch(self, batch: Batch, metrics: MetricTracker, to_log: bool):
        batch.to(self.device)
        self.optimizer.zero_grad()
        outputs, new_lns, pred_log_len, true_log_len = self.model(batch, self.device, self.criterion_fs.melspec, self.galigner)

        loss_fs = self.criterion_fs(outputs, new_lns, batch)
        loss_dp = self.criterion_dp(batch, pred_log_len, true_log_len)

        loss = loss_fs + loss_dp
        loss.backward()
        self._clip_grad_norm()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if to_log:
            j = random.randint(0, outputs.size(0)-1)
            ground_truth_melspec = self.criterion_fs.melspec(batch.waveform)
            self._log_spectrogram("train_pred", outputs[j].detach())
            self._log_spectrogram("train_ground_truth", ground_truth_melspec[j])

        metrics.update("loss_fs", loss_fs.item())
        metrics.update("loss_dp", loss_dp.item())

        return loss_fs.item(), loss_dp.item()

    def _valid_example(self, n_examples=1):
        """
        see how model works on example
        """
        self.model.eval()
        with torch.no_grad():
            for i in range(n_examples):
                batch = next(iter(self.val_data_loader))
                batch.to(self.device)
                ground_truth_melspec = self.criterion_fs.melspec(batch.waveform)
                output = self.model(batch, self.device, self.criterion_fs.melspec, self.galigner)
                # output: [1, sq_len, 80]
                pred_wav = self.vocoder.inference(output.transpose(-1, -2)).cpu()
                true_wav = self.vocoder.inference(ground_truth_melspec.transpose(-1, -2)).cpu()

                self._log_spectrogram("val_pred", output[0])
                self._log_spectrogram("val_ground_truth", ground_truth_melspec[0])

                self._log_audios("val_pred_synth", pred_wav.squeeze())
                self._log_audios("val_true_synth", true_wav.squeeze())

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #    self.writer.add_histogram(name, p, bins="auto")

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_spectrogram(self, name, spec):
        image = PIL.Image.open(plot_spectrogram_to_buf(spec.cpu()))
        self.writer.add_image(name, (ToTensor()(image)).transpose(-1, -2).flip(-2))

    def _log_audios(self, name, audio_example):
        audio = audio_example
        self.writer.add_audio(name, audio, sample_rate=self.config["MelSpectrogram"]["sr"])

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
