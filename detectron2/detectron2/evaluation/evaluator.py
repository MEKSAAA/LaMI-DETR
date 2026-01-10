# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], debug=False
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            if debug:
                if idx == 15:
                    print("BREAK!"*5)
                    break
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


def inference_on_roi_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], debug=False,
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        groundtruth = []
        prediction = []
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            if inputs[0]['instances'] == None:
                continue
            else:
                pred_out = model(inputs)

            prediction.extend(pred_out.tolist())
            groundtruth.extend(inputs[0]['instances'].gt_classes.tolist())

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            # evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            if debug:
                if idx == 15:
                    print("BREAK!"*5)
                    break
            start_data_time = time.perf_counter()
    

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    # add macro weighted acc metric
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_score,confusion_matrix
    import torch.distributed as dist
    import itertools
    import detectron2.utils.comm as comm
    if dist.is_initialized():
        comm.synchronize()
        predictions = comm.gather(prediction, dst=0)
        predictions = list(itertools.chain(*predictions))
        groundtruths = comm.gather(groundtruth, dst=0)
        groundtruths = list(itertools.chain(*groundtruths))
        if not comm.is_main_process():
            return {}
    else:
        predictions = prediction
        groundtruths = groundtruth

    if max(predictions) > 100:
        num_class = 1203
        all_labels = [i for i in range(num_class)]
        confusion_mat = confusion_matrix(groundtruths, predictions, labels=all_labels)
        acc_score = []
        for i in range(confusion_mat.shape[0]):
            if confusion_mat[i, :].sum() == 0: 
                class_accuracy = -1
            else:
                class_accuracy = confusion_mat[i, i] / confusion_mat[i, :].sum()
            acc_score.append(class_accuracy)
        
        import json
        with open("dataset/lvis/lvis_v1_val.json", "r") as file:
            lvis = json.load(file)
        freq_groups = {'f':[], 'c':[], 'r':[], 'vgr':[]}
        for idx, _cat_data in enumerate(lvis["categories"]):
            freq_groups[_cat_data['frequency']].append(idx)
        freq_groups['vgr'] = [0, 6, 13, 17, 21, 24, 26, 27, 30, 37, 39, 40, 41, 43, 44, 45, 52, 55, 56, 61, 63, 68, 69, 77, 80, 86, 87, 91, 94, 96, 105, 106, 112, 115, 116, 121, 122, 124, 130, 135, 136, 138, 139, 141, 145, 146, 156, 157, 158, 161, 166, 179, 183, 184, 187, 192, 200, 203, 204, 208, 209, 210, 211, 213, 214, 215, 219, 223, 225, 228, 230, 235, 238, 240, 244, 246, 251, 252, 254, 255, 257, 259, 261, 262, 264, 266, 268, 269, 274, 275, 280, 282, 286, 290, 292, 293, 298, 308, 314, 320, 322, 328, 333, 337, 339, 340, 344, 348, 350, 351, 354, 356, 359, 369, 373, 378, 382, 383, 385, 390, 394, 395, 396, 397, 399, 401, 404, 406, 414, 417, 418, 419, 431, 434, 435, 438, 445, 447, 448, 451, 457, 458, 463, 465, 471, 478, 480, 481, 485, 491, 494, 498, 505, 508, 509, 513, 516, 517, 526, 529, 530, 531, 534, 542, 556, 565, 567, 569, 570, 572, 577, 581, 583, 586, 587, 593, 598, 601, 605, 615, 617, 625, 629, 632, 637, 640, 645, 646, 653, 657, 661, 664, 665, 669, 672, 673, 674, 676, 679, 680, 687, 689, 692, 695, 696, 707, 708, 710, 711, 716, 722, 725, 729, 731, 734, 735, 736, 741, 742, 743, 744, 749, 751, 754, 768, 769, 770, 773, 776, 777, 786, 787, 790, 791, 804, 805, 807, 808, 818, 820, 822, 829, 831, 832, 841, 844, 845, 846, 850, 853, 854, 858, 864, 865, 866, 868, 870, 871, 873, 874, 876, 882, 884, 885, 886, 888, 892, 893, 894, 896, 901, 906, 907, 914, 918, 929, 932, 933, 937, 938, 939, 941, 944, 945, 967, 968, 970, 971, 973, 983, 984, 988, 989, 991, 992, 997, 1001, 1014, 1015, 1018, 1023, 1028, 1040, 1043, 1048, 1053, 1054, 1062, 1067, 1074, 1083, 1087, 1094, 1104, 1109, 1115, 1118, 1123, 1124, 1127, 1129, 1139, 1142, 1146, 1152, 1170, 1171, 1179, 1180, 1193, 1200, 1202]

        for freq_group in ['f', 'c', 'r', 'vgr']:
            index = freq_groups[freq_group]
            score = [acc_score[i] for i in index if acc_score[i] != -1]
            log_score = sum(score)/len(score) if len(score) !=0 else 0
            logger.info(f"{freq_group}: {log_score * 100:.3f}")
    else:
        # import ipdb;ipdb.set_trace()
        num_class = 65
        all_labels = [i for i in range(num_class)]
        confusion_mat = confusion_matrix(groundtruths, predictions, labels=all_labels)
        acc_score = []
        for i in range(confusion_mat.shape[0]):
            if confusion_mat[i, :].sum() == 0: 
                class_accuracy = -1
            else:
                class_accuracy = confusion_mat[i, i] / confusion_mat[i, :].sum()
            acc_score.append(class_accuracy)
        # import ipdb;ipdb.set_trace()
        unseen_index = [4, 5, 11, 12, 15, 16, 21, 23, 27, 29, 32, 34, 45, 47, 54, 58, 63]
        seen_index = [i for i in range(65) if i not in unseen_index]
        unseen_score = [acc_score[i] for i in unseen_index if acc_score[i] != -1]
        log_score = sum(unseen_score)/len(unseen_score) if len(unseen_score) !=0 else 0
        logger.info(f"unseen: {log_score * 100:.3f}")
        seen_score = [acc_score[i] for i in seen_index if acc_score[i] != -1]
        log_score = sum(seen_score)/len(seen_score) if len(seen_score) !=0 else 0
        logger.info(f"seen: {log_score * 100:.3f}")

    macro_mAP = precision_score(groundtruths, predictions, average="macro")
    weighted_mAP = precision_score(groundtruths, predictions, average="weighted")
    acc = accuracy_score(groundtruths, predictions)
    macc = balanced_accuracy_score(groundtruths, predictions)
    logger.info(
        f"macro_mAP, weighted_mAP, Acc, mAcc: {macro_mAP * 100:.3f} {weighted_mAP * 100:.3f} {acc * 100:.3f} {macc * 100:.3f}"
    )
    
    return {}


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
