"""
Benchmark script for Parakeet STT API
Evaluates WER and performance metrics across different precision levels
"""
import argparse
import csv
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import soundfile as sf

import config
from backend import create_backend


def normalize_text(text: str) -> str:
    """
    Normalize text for WER calculation
    - Strip markup tags and action markers
    - Convert to lowercase
    - Normalize contractions and variations
    - Remove punctuation
    - Normalize whitespace
    """
    # Strip HTML-like tags: <tag>content</tag> or <tag#...>content</tag#...>
    text = re.sub(r'<[^>]+>', '', text)

    # Strip curly brace markup: {RUBY_B#...}text{RUBY_E#...}
    text = re.sub(r'\{[^}]+\}', '', text)

    # Strip action markers: *action*
    text = re.sub(r'\*[^*]+\*', '', text)

    # Convert to lowercase
    text = text.lower()

    # Handle stutters: "b—but" -> "but", "s—seele" -> "seele"
    # Pattern: single letter + em-dash + same letter starting word
    text = re.sub(r'\b([a-z])—\1', r'\1', text)

    # Normalize remaining em-dashes and multiple dashes to single space
    text = re.sub(r'[—–-]+', ' ', text)

    # Normalize common contractions to expanded forms
    contractions = {
        r'\bgonna\b': 'going to',
        r'\bwanna\b': 'want to',
        r"\bc'mon\b": 'come on',
        r"\bcmon\b": 'come on',
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)

    # Normalize common spelling variations
    variations = {
        r'\balright\b': 'all right',
        r'\bmadame\b': 'madam',
    }
    for pattern, replacement in variations.items():
        text = re.sub(pattern, replacement, text)

    # Remove all remaining punctuation (keep only alphanumeric and spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # Normalize whitespace
    text = ' '.join(text.split())

    return text


def calculate_wer(reference: str, hypothesis: str) -> Tuple[float, int, int, int, int]:
    """
    Calculate Word Error Rate (WER)

    Returns:
        wer: Word Error Rate (0.0 to 1.0+)
        substitutions: Number of word substitutions
        deletions: Number of word deletions
        insertions: Number of word insertions
        total_words: Total words in reference
    """
    # Normalize both texts
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    # Calculate edit distance using dynamic programming
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Initialize
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,    # deletion
                    dp[i][j-1] + 1,    # insertion
                    dp[i-1][j-1] + 1   # substitution
                )

    # Backtrack to count error types
    i, j = n, m
    substitutions = 0
    deletions = 0
    insertions = 0

    while i > 0 or j > 0:
        if i == 0:
            insertions += 1
            j -= 1
        elif j == 0:
            deletions += 1
            i -= 1
        elif ref_words[i-1] == hyp_words[j-1]:
            i -= 1
            j -= 1
        else:
            # Find which operation was used
            min_val = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            if dp[i-1][j-1] == min_val:
                substitutions += 1
                i -= 1
                j -= 1
            elif dp[i-1][j] == min_val:
                deletions += 1
                i -= 1
            else:
                insertions += 1
                j -= 1

    total_errors = substitutions + deletions + insertions
    total_words = len(ref_words)
    wer = total_errors / total_words if total_words > 0 else 0.0

    return wer, substitutions, deletions, insertions, total_words


def get_vram_usage():
    """Get current VRAM usage in MB (returns None if not available)"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
    except:
        pass
    return None


def load_dataset(dataset_path: Path) -> List[Tuple[Path, str]]:
    """
    Load dataset of wav files and their corresponding text files

    Returns:
        List of (audio_path, ground_truth_text) tuples
    """
    dataset = []

    # Find all .wav files
    wav_files = sorted(dataset_path.glob("*.wav"))

    for wav_file in wav_files:
        # Find corresponding .txt file
        txt_file = wav_file.with_suffix('.txt')

        if not txt_file.exists():
            print(f"Warning: Missing ground truth for {wav_file.name}, skipping...")
            continue

        # Read ground truth text
        with open(txt_file, 'r', encoding='utf-8') as f:
            ground_truth = f.read().strip()

        dataset.append((wav_file, ground_truth))

    print(f"Loaded {len(dataset)} files from {dataset_path}")
    return dataset


def benchmark_precision(
    precision: str,
    dataset: List[Tuple[Path, str]],
    num_iterations: int,
    force_cpu: bool = False,
    debug_log_file: Path = None,
    num_threads: int = 0
) -> Dict:
    """
    Benchmark a specific precision level

    Args:
        precision: Precision level to test
        dataset: List of (audio_path, ground_truth) tuples
        num_iterations: Number of iterations to run
        force_cpu: Force CPU mode (default: False for GPU if available)
        debug_log_file: Optional path to write detailed error log
        num_threads: Number of CPU threads (0 = auto-detect)

    Returns:
        Dictionary with metrics: wer, rtf, total_time, vram, etc.
    """
    device_mode = "CPU (forced)" if force_cpu else "GPU/CPU (auto)"
    print(f"\n{'='*70}")
    print(f"Benchmarking {precision.upper()} - {device_mode}")
    print(f"{'='*70}")

    # Initialize model
    model_dir = None
    if precision == 'int8':
        model_dir_name = config.ONNX_MODEL_DIRS.get(precision)
        if model_dir_name:
            model_dir = config.MODELS_DIR / model_dir_name

    print(f"Loading {precision.upper()} model...")
    backend = create_backend(
        precision=precision,
        model_dir=model_dir,
        num_threads=num_threads,
        force_cpu=force_cpu
    )
    print(f"✓ Model loaded: {backend.backend_name}")

    # Track VRAM after model load
    vram_after_load = get_vram_usage()

    # Open debug log file if specified
    debug_file = None
    if debug_log_file:
        debug_file = open(debug_log_file, 'w', encoding='utf-8')
        debug_file.write(f"# Debug Log: {precision.upper()} - {device_mode}\n")
        debug_file.write(f"# Format: Each error shows reference vs hypothesis (normalized text used for WER calculation)\n")
        debug_file.write("=" * 80 + "\n\n")

    # Run benchmark iterations
    all_wers = []
    all_rtfs = []
    all_times = []
    all_errors = {'substitutions': 0, 'deletions': 0, 'insertions': 0, 'total_words': 0}

    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")

        iteration_wers = []
        iteration_rtfs = []
        iteration_time = 0

        for audio_path, ground_truth in dataset:
            # Load audio
            audio, sr = sf.read(str(audio_path), dtype='float32')

            # Convert stereo to mono if needed
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # Resample if needed
            if sr != config.DEFAULT_SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=config.DEFAULT_SAMPLE_RATE)

            # Calculate audio duration
            audio_duration = len(audio) / config.DEFAULT_SAMPLE_RATE

            # Transcribe with timing
            start_time = time.time()
            result = backend.transcribe(audio)
            inference_time = time.time() - start_time

            hypothesis = result['text']

            # Calculate WER
            wer, subs, dels, ins, total_words = calculate_wer(ground_truth, hypothesis)

            # Calculate RTF
            rtf = inference_time / audio_duration

            # Write to debug log if there are errors
            if debug_file and wer > 0:
                # Get normalized versions (as used in WER calculation)
                ref_normalized = normalize_text(ground_truth)
                hyp_normalized = normalize_text(hypothesis)

                debug_file.write(f"File: {audio_path.name}\n")
                debug_file.write(f"WER: {wer:.4f} (Subs: {subs}, Dels: {dels}, Ins: {ins})\n")
                debug_file.write(f"\nReference (original):\n  {ground_truth}\n")
                debug_file.write(f"\nReference (normalized):\n  {ref_normalized}\n")
                debug_file.write(f"\nHypothesis (original):\n  {hypothesis}\n")
                debug_file.write(f"\nHypothesis (normalized):\n  {hyp_normalized}\n")
                debug_file.write("\n" + "-" * 80 + "\n\n")
                debug_file.flush()

            # Accumulate metrics
            iteration_wers.append(wer)
            iteration_rtfs.append(rtf)
            iteration_time += inference_time

            all_errors['substitutions'] += subs
            all_errors['deletions'] += dels
            all_errors['insertions'] += ins
            all_errors['total_words'] += total_words

            # Print progress
            print(f"  {audio_path.name}: WER={wer:.4f}, RTF={rtf:.4f}")

        # Store iteration averages
        all_wers.extend(iteration_wers)
        all_rtfs.extend(iteration_rtfs)
        all_times.append(iteration_time)

        print(f"  Iteration avg: WER={np.mean(iteration_wers):.4f}, RTF={np.mean(iteration_rtfs):.4f}")

    # Calculate final averages
    avg_wer = np.mean(all_wers)
    avg_rtf = np.mean(all_rtfs)
    avg_time = np.mean(all_times)

    # Calculate total WER from accumulated errors
    total_wer = (all_errors['substitutions'] + all_errors['deletions'] + all_errors['insertions']) / all_errors['total_words'] if all_errors['total_words'] > 0 else 0.0

    results = {
        'precision': precision.upper(),
        'wer': avg_wer,
        'total_wer': total_wer,
        'rtf': avg_rtf,
        'total_time': avg_time,
        'substitutions': all_errors['substitutions'] / num_iterations,
        'deletions': all_errors['deletions'] / num_iterations,
        'insertions': all_errors['insertions'] / num_iterations,
        'vram_mb': vram_after_load,
    }

    print(f"\n{precision.upper()} Results:")
    print(f"  Average WER: {avg_wer:.4f}")
    print(f"  Total WER: {total_wer:.4f}")
    print(f"  Average RTF: {avg_rtf:.4f}")
    print(f"  Average Time: {avg_time:.2f}s")
    if vram_after_load:
        print(f"  VRAM Usage: {vram_after_load:.2f} MB")

    # Close debug log file
    if debug_file:
        debug_file.close()
        print(f"  Debug log written to: {debug_log_file}")

    # Clean up model to free VRAM before next precision test
    del backend

    # Force garbage collection and clear CUDA cache
    import gc
    gc.collect()

    # Clear PyTorch CUDA cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Parakeet STT across different precision levels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark with all precisions
  python benchmark.py --dataset ./test_data --iterations 3 --output results.csv

  # Test both CPU and GPU for each precision
  python benchmark.py --dataset ./test_data --device both --output results.csv

  # Test only on CPU
  python benchmark.py --dataset ./test_data --device cpu --output results.csv

  # Generate debug log showing transcription errors
  python benchmark.py --dataset ./test_data --debug-log errors.txt

  # Test both devices with debug logs
  python benchmark.py --dataset ./test_data --device both --debug-log errors.txt

Dataset format:
  test_data/
    audio1.wav
    audio1.txt
    audio2.wav
    audio2.txt
    ...
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to dataset directory containing .wav and .txt files'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=1,
        help='Number of iterations to run for averaging (default: 1)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_results.csv',
        help='Output CSV file (default: benchmark_results.csv)'
    )
    parser.add_argument(
        '--precisions',
        nargs='+',
        choices=['fp32', 'int8'],
        default=['fp32', 'int8'],
        help='Precision levels to benchmark (default: all)'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU usage even if GPU is available'
    )
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu', 'both'],
        default=None,
        help='Device to test on: cpu, gpu, or both (default: auto based on --cpu flag)'
    )
    parser.add_argument(
        '--debug-log',
        type=str,
        default=None,
        help='Write detailed error log to this file (shows reference vs hypothesis for errors)'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=config.DEFAULT_NUM_THREADS,
        help=f'Number of CPU threads, 0=auto-detect physical cores (default: {config.DEFAULT_NUM_THREADS})'
    )

    args = parser.parse_args()

    # Handle device selection
    if args.device:
        test_devices = []
        if args.device == 'cpu' or args.device == 'both':
            test_devices.append(('CPU', True))
        if args.device == 'gpu' or args.device == 'both':
            test_devices.append(('GPU', False))
    else:
        # Default: use --cpu flag behavior
        test_devices = [('CPU' if args.cpu else 'Auto', args.cpu)]

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        return

    dataset = load_dataset(dataset_path)
    if not dataset:
        print("Error: No valid audio/text pairs found in dataset")
        return

    print(f"\nBenchmark Configuration:")
    print(f"  Dataset: {dataset_path}")
    print(f"  Files: {len(dataset)}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Precisions: {', '.join(args.precisions)}")
    print(f"  Devices: {', '.join([d[0] for d in test_devices])}")
    if args.threads == 0:
        import os
        try:
            import psutil
            detected_threads = psutil.cpu_count(logical=False) or (os.cpu_count() // 2) or 4
        except ImportError:
            detected_threads = (os.cpu_count() // 2) or 4
        print(f"  CPU Threads: {detected_threads} physical cores (auto-detected)")
    else:
        print(f"  CPU Threads: {args.threads}")
    print(f"  Output: {args.output}")
    if args.debug_log:
        print(f"  Debug Log: {args.debug_log}")

    # Run benchmarks
    results = []
    fp32_gpu_wer = None

    for precision in args.precisions:
        for device_name, force_cpu in test_devices:
            # Skip GPU test for INT8 (INT8 is CPU-only)
            if precision == 'int8' and not force_cpu:
                print(f"\nSkipping {precision.upper()} GPU test (INT8 is CPU-only)")
                continue

            # Generate debug log filename if requested
            debug_log = None
            if args.debug_log:
                base_name = Path(args.debug_log).stem
                ext = Path(args.debug_log).suffix
                debug_log = Path(f"{base_name}_{precision}_{device_name.lower()}{ext}")

            result = benchmark_precision(
                precision,
                dataset,
                args.iterations,
                force_cpu=force_cpu,
                debug_log_file=debug_log,
                num_threads=args.threads
            )

            # Add device info to result
            result['device'] = device_name

            # Calculate delta from FP32 GPU baseline
            if precision == 'fp32' and not force_cpu:
                fp32_gpu_wer = result['wer']
                result['delta_from_fp32_gpu'] = 0.0
            else:
                if fp32_gpu_wer is not None:
                    result['delta_from_fp32_gpu'] = result['wer'] - fp32_gpu_wer
                else:
                    result['delta_from_fp32_gpu'] = None

            results.append(result)

    # Write results to CSV
    print(f"\n{'='*70}")
    print(f"Writing results to {args.output}")
    print(f"{'='*70}")

    with open(args.output, 'w', newline='') as f:
        fieldnames = [
            'precision',
            'device',
            'wer',
            'total_wer',
            'rtf',
            'total_time',
            'delta_from_fp32_gpu',
            'substitutions',
            'deletions',
            'insertions',
            'vram_mb'
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(result)

    # Print summary table
    print("\nBenchmark Summary:")
    print("-" * 120)
    print(f"{'Precision':<12} {'Device':<8} {'WER':<10} {'RTF':<10} {'Time (s)':<12} {'Delta WER':<12} {'VRAM (MB)':<12}")
    print("-" * 120)

    for result in results:
        vram_str = f"{result['vram_mb']:.2f}" if result['vram_mb'] else "N/A"
        delta_str = f"{result['delta_from_fp32_gpu']:+.4f}" if result['delta_from_fp32_gpu'] is not None else "N/A"

        print(f"{result['precision']:<12} {result['device']:<8} {result['wer']:<10.4f} {result['rtf']:<10.4f} "
              f"{result['total_time']:<12.2f} {delta_str:<12} {vram_str:<12}")

    print("-" * 120)
    print(f"\nResults saved to: {args.output}")
    if args.debug_log:
        print(f"Debug logs saved with pattern: {Path(args.debug_log).stem}_<precision>_<device>{Path(args.debug_log).suffix}")


if __name__ == "__main__":
    main()
