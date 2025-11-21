import torch
import torch.nn as nn
import numpy as np
import os

from record import PerformanceMeter
from utils import count_parameters






class Trainer(nn.Module):
    r'''A Multi-Task Learning Trainer.
    Assumes data from DataLoader is already collated (by DataCollator as collate_fn)
    and is a dictionary of tensors.
    Model's forward pass is now expected to return a single value:
    - A dictionary of predictions for 'train' (final step) and 'val'/'test' modes.
    - None for 'train' mode intermediate steps (encoder accumulating).
    '''

    def __init__(self, task_dict, weighting, architecture, encoder_class, decoders,
                 optim_param, args, save_path = None, load_path = None, **kwargs):
        super(Trainer, self).__init__()

        if args.gpu_id != 'cpu' and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{args.gpu_id}')
        else:
            self.device = torch.device('cpu')

        self.kwargs = kwargs
        self.task_dict = task_dict
        self.task_num = len(task_dict)
        self.task_name = list(task_dict.keys())  # Trainer's list of tasks
        self.save_path = save_path
        self.load_path = load_path
        self.args = args

        self.prepare_model(weighting, architecture, encoder_class, decoders)
        self.prepare_optimizer(optim_param)
        self.meter = PerformanceMeter(self.task_dict)

    def prepare_model(self, weighting, architecture, encoder_class, decoders):
        class MTLmodel(architecture, weighting):
            def __init__(self, task_name_list, enc_class, dec_dict, dev, cmd_args, arch_kwargs_val):
                super(MTLmodel, self).__init__(task_name_list, enc_class, dec_dict, dev, cmd_args, **arch_kwargs_val)
                if hasattr(self, 'init_param'):
                    self.init_param()
                else:
                    print(
                        f"DEBUG: Model class {type(self).__name__} (bases: {type(self).__bases__}) does not have an init_param method.")

        self.model = MTLmodel(task_name_list = self.task_name,
                              enc_class = encoder_class,
                              dec_dict = decoders,
                              dev = self.device,
                              cmd_args = self.args,
                              arch_kwargs_val = self.kwargs.get('arch_args', {})
                              ).to(self.device)



        if self.load_path is not None:
            if os.path.isdir(self.load_path):
                ckpt_file_name = f'{self.args.ckpt_name}_best.pt' if hasattr(self.args, 'ckpt_name') else 'best.pt'
                self.load_path = os.path.join(self.load_path, ckpt_file_name)

            if os.path.exists(self.load_path):
                try:
                    self.model.load_state_dict(torch.load(self.load_path, map_location = self.device), strict = False)
                    print('Successfully loaded model from - {}'.format(self.load_path))
                except Exception as e:
                    print(
                        f"Error loading model from {self.load_path}: {e}. Training from scratch or with initial weights.")
            else:
                print(f"Model file not found at {self.load_path}. Training from scratch or with initial weights.")
        count_parameters(self.model)

    def prepare_optimizer(self, optim_param):
        optim_dict = {'adam': torch.optim.Adam}
        optim_name = optim_param.get('optim', 'adam').lower()
        if optim_name not in optim_dict:
            raise ValueError(f"Unsupported optimizer: {optim_name}")
        optim_arg = {k: v for k, v in optim_param.items() if k != 'optim'}
        self.optimizer = optim_dict[optim_name](self.model.parameters(), **optim_arg)

    def compute_loss(self, preds, gts, task_name = None):
        loss = self.meter.losses[task_name].update_loss(preds, gts.to(preds.device))
        return loss

    def _prepare_iterators_and_counts(self, dataloaders_dict):
        iterators = {}
        counts = {}
        if dataloaders_dict:
            for task, loader in dataloaders_dict.items():
                if loader:  # Check if loader itself is not None
                    if len(loader) > 0:
                        iterators[task] = iter(loader)
                        counts[task] = len(loader)
                    else:  # Loader exists but is empty
                        iterators[task] = None  # No iterator for empty loader
                        counts[task] = 0
                        print(f"Warning: DataLoader for task '{task}' is empty (length 0).")
                else:  # Loader is None
                    counts[task] = 0
                    iterators[task] = None
                    print(f"Warning: DataLoader for task '{task}' is None.")
        return iterators, counts

    def train(self, train_dataloaders_dict, val_dataloaders_dict, test_dataloaders_dict, epochs, params_main):
        train_iterators, train_batch_counts = self._prepare_iterators_and_counts(train_dataloaders_dict)

        # Determine max_batches_per_epoch based on tasks that actually have data
        active_train_batch_counts = [count for task, count in train_batch_counts.items() if
                                     train_iterators.get(task) is not None]
        if not active_train_batch_counts:
            print("Error: All training dataloaders are effectively empty. Cannot start training.")
            return
        max_batches_per_epoch = max(active_train_batch_counts) if active_train_batch_counts else 0
        if max_batches_per_epoch == 0:
            print("Error: max_batches_per_epoch is 0. All tasks might have empty dataloaders. Cannot start training.")
            return

        if hasattr(self.model, 'train_loss_buffer'): self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        if hasattr(self.model, 'epochs'): self.model.epochs = epochs
        self.batch_weight = np.zeros([self.task_num, epochs, max_batches_per_epoch])

        for epoch in range(epochs):
            if hasattr(self.model, 'epoch'): self.model.epoch = epoch
            self.model.train()
            self.meter.record_time('begin')

            for batch_idx in range(max_batches_per_epoch):
                current_epoch_batch_step_losses = torch.zeros(self.task_num, device = self.device)
                ground_truths_for_step = {}
                active_tasks_in_step = []  # Tasks that successfully provided data for this batch_idx
                model_output_predictions_final_for_step = None

                for task_order_index, current_task_name in enumerate(self.task_name):
                    batch_data_for_task = None
                    current_task_iterator = train_iterators.get(current_task_name)

                    if current_task_iterator is None:  # Task had an empty dataloader initially

                        if epoch == 0 and batch_idx == 0:  # Print warning once
                            print(
                                f"Warning: Task {current_task_name} has an empty DataLoader. It will be skipped in model input feeding.")
                        continue

                    try:
                        batch_data_for_task = next(current_task_iterator)
                    except StopIteration:

                        if not train_dataloaders_dict.get(current_task_name) or \
                                len(train_dataloaders_dict[current_task_name]) == 0:

                            print(
                                f"ERROR: Task {current_task_name} DataLoader is unexpectedly None or empty upon StopIteration.")
                            continue

                        if epoch == 0 and batch_idx < 2:  # Print only for first few batches of first epoch for brevity
                            print(
                                f"INFO: Re-initializing iterator for task {current_task_name} at epoch {epoch}, batch {batch_idx}")

                        train_iterators[current_task_name] = iter(train_dataloaders_dict[current_task_name])
                        try:
                            batch_data_for_task = next(train_iterators[current_task_name])
                        except StopIteration:
                            print(f"ERROR: Task {current_task_name} failed to yield data even after re-initializing. "
                                  f"Its dataset might be truly empty or too small for batch_size {self.args.bs}. Skipping this task for this step.")
                            continue  # Skip if still no data


                    if batch_data_for_task is None:
                        continue

                    if batch_data_for_task.get('is_empty', False):

                        if epoch == 0 and batch_idx == 0:  # Print warning once
                            print(
                                f"Warning: Collator returned an empty batch for task {current_task_name} at epoch {epoch}, batch {batch_idx}. Skipping model call for this task's data.")

                        continue

                    ground_truths_for_step[current_task_name] = batch_data_for_task['y']
                    if current_task_name not in active_tasks_in_step:
                        active_tasks_in_step.append(current_task_name)

                    # Model now returns one value: preds dict or None
                    temp_preds = self.model(batch_data_for_task, current_task_name, 'train')

                    if temp_preds is not None:  # Encoder finished its cycle
                        model_output_predictions_final_for_step = temp_preds

                # After iterating all tasks for model input accumulation for this batch_idx:
                if model_output_predictions_final_for_step is not None:
                    losses_for_backward_pass = []
                    for loss_calc_idx, task_name_for_loss in enumerate(self.task_name):
                        # Only calculate loss if task was active, has prediction, and has GT
                        if task_name_for_loss in active_tasks_in_step and \
                                task_name_for_loss in model_output_predictions_final_for_step and \
                                task_name_for_loss in ground_truths_for_step:
                            pred_tensor = model_output_predictions_final_for_step[task_name_for_loss]
                            gt_tensor = ground_truths_for_step[task_name_for_loss]

                            loss_val = self.compute_loss(pred_tensor, gt_tensor, task_name_for_loss)
                            current_epoch_batch_step_losses[loss_calc_idx] = loss_val
                            losses_for_backward_pass.append(loss_val)
                            self.meter.update(pred_tensor, gt_tensor, task_name_for_loss)  # Updates metrics
                        # else: current_epoch_batch_step_losses[loss_calc_idx] remains 0.0 if task was not active or had no prediction

                    if not losses_for_backward_pass:  # No valid losses to backprop
                        if batch_idx == 0 and epoch == 0 and active_tasks_in_step:  # Print only once for brevity
                            print(f"Warning: Train Batch {batch_idx}: Active tasks were {active_tasks_in_step}, "
                                  f"but no losses computed. Predictions: "
                                  f"{model_output_predictions_final_for_step.keys() if model_output_predictions_final_for_step else 'None'}. "
                                  f"Ground truths for: {ground_truths_for_step.keys()}. Skipping backward pass.")
                        continue

                    self.optimizer.zero_grad()
                    output_weights = self.model.backward(current_epoch_batch_step_losses,
                                                         **self.kwargs.get('weight_args', {}))
                    if output_weights is not None:
                        self.batch_weight[:, epoch, batch_idx] = output_weights.cpu().numpy() if isinstance(
                            output_weights, torch.Tensor) else output_weights
                    self.optimizer.step()

                elif active_tasks_in_step:  # Data was fed, but model didn't produce final output this step
                    if epoch == 0 and batch_idx == 0:  # Print warning only once for brevity
                        print(
                            f"Warning: Train Batch {batch_idx}: Model processed inputs from active tasks ({active_tasks_in_step}) "
                            f"but did not return final predictions. This is expected if the encoder's full task sequence "
                            f"for this meta-batch was not completed (e.g. if last task in encoder sequence had no data this batch_idx).")

            # --- End of epoch ---
            self.meter.record_time('end')
            self.meter.get_score()
            if hasattr(self.model, 'train_loss_buffer'): self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
            self.meter.display(mode = 'train', epoch = epoch)
            self.meter.reinit()

            if val_dataloaders_dict and any(val_dataloaders_dict.values()): self.test(val_dataloaders_dict,
                                                                                      epoch = epoch, mode = 'val')
            if test_dataloaders_dict and any(test_dataloaders_dict.values()): self.test(test_dataloaders_dict,
                                                                                        epoch = epoch, mode = 'test')

            if self.save_path is not None:
                is_best_epoch = False
                # Assuming PerformanceMeter's best_val_epoch['average'][0] holds the epoch number of the best average validation metric
                avg_best_epoch_info = self.meter.best_val_epoch.get('average')
                if avg_best_epoch_info and avg_best_epoch_info[0] == epoch:
                    is_best_epoch = True

                if is_best_epoch:
                    model_save_file = os.path.join(self.save_path, f'{params_main.ckpt_name}_best.pt')
                    torch.save(self.model.state_dict(), model_save_file)
                    print(f'Saved Best Model (epoch {epoch}) based on validation to {model_save_file}')

        self.meter.display_best_result()

    def test(self, dataloaders_dict, epoch = None, mode = 'test', visualize_prompts = False, visualize_predictions = False,
             visualize_histograms=False, args_for_viz = None):
        test_iterators, test_batch_counts = self._prepare_iterators_and_counts(dataloaders_dict)
        if not any(test_batch_counts.values()):
            print(f"No data for {mode} mode. Skipping.")
            return

        self.model.eval()
        self.meter.record_time('begin')
        with torch.no_grad():
            for task_order_index, task_name in enumerate(self.task_name):
                current_task_iterator = test_iterators.get(task_name)
                current_task_batch_count = test_batch_counts.get(task_name, 0)

                if not current_task_iterator or current_task_batch_count == 0:
                    # print(f"DEBUG: Skipping task {task_name} in {mode} mode as it has no data.")
                    continue

                if hasattr(self.meter, 'cache_result') and task_name in self.meter.cache_result:
                    self.meter.cache_result[task_name]['pred'] = []
                    self.meter.cache_result[task_name]['gts'] = []

                task_processed_at_least_one_batch = False
                for batch_idx in range(current_task_batch_count):
                    try:
                        batch_data = next(
                            current_task_iterator)  # In test mode, we don't re-initialize, just iterate through once
                        if batch_data.get('is_empty', False): continue

                        predictions_output = self.model(batch_data, task_name, mode)  # Model returns dict for this task

                        if predictions_output is not None and task_name in predictions_output:
                            current_task_pred = predictions_output[task_name]
                            current_task_gt = batch_data['y']
                            self.meter.append_result(current_task_pred, current_task_gt, task_name)
                            task_processed_at_least_one_batch = True
                        else:
                            print(
                                f"Warning: No prediction output for task {task_name} in {mode} mode, batch {batch_idx}.")
                    except StopIteration:  # Should ideally not happen if looping up to test_batch_counts
                        print(f"Warning: StopIteration unexpected in {mode} for task {task_name} at batch {batch_idx}.")
                        break
                    except Exception as e:
                        print(f"Error during {mode} for task {task_name}, batch {batch_idx}: {e}")
                        continue

                if task_processed_at_least_one_batch and \
                        hasattr(self.meter, 'cache_result') and \
                        self.meter.cache_result.get(task_name) and \
                        self.meter.cache_result[task_name]['pred'] and \
                        self.meter.cache_result[task_name]['gts']:

                    pred_all_for_task = torch.cat(self.meter.cache_result[task_name]['pred'], dim = 0)
                    gts_all_for_task = torch.cat(self.meter.cache_result[task_name]['gts'], dim = 0)

                    self.compute_loss(pred_all_for_task, gts_all_for_task, task_name)  # Record loss for val/test
                    self.meter.update(pred_all_for_task, gts_all_for_task, task_name)  # Update metrics
                elif not task_processed_at_least_one_batch and current_task_batch_count > 0:
                    print(
                        f"Warning: No batches successfully processed for task {task_name} in {mode} mode, though loader had data.")

        self.meter.record_time('end')
        self.meter.get_score()
        self.meter.display(mode = mode, epoch = epoch)




        self.meter.reinit()
