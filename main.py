"""
Main real-time evolution loop for isolated OLA genome visualization

Process:
1. Generate random input
2. VAE encodes to latent vector
3. PatternLSTM extracts temporal features
4. All genomes process the pattern vector
5. Genomes mutate based on trust scores
6. Visualize population in real-time
"""
import torch
import numpy as np
import time
import argparse
import psutil
import csv
import os
import threading
import math
def worker_job(stop_event: threading.Event, intensity: int):
    """Synthetic CPU workload controlled by feedback intensity"""
    x = 0.0
    while not stop_event.is_set():
        for _ in range(int(intensity)):
            x += math.sin(x)
        # Always yield briefly to avoid starving the main thread
        time.sleep(0.001)


from vae import SimpleVAE
from pattern_lstm import PatternLSTM
from genome_library import GenomeLibrary
from lsh_genome_database import LSHGenomeDatabase
from visualizer import GenomeVisualizer


class IsolatedOLA:
    """
    Self-contained evolution visualizer
    """

    def __init__(self, device: str = "cpu", visualize: bool = True):
        """
        Initialize the isolated OLA system

        Args:
            device: Device to run on ("cpu" or "cuda")
            visualize: Whether to show visualization
        """
        self.device = device
        self.visualize = visualize

        # Dimensions
        self.input_dim = 128
        self.vae_latent_dim = 32
        self.pattern_dim = 32
        self.genome_out_dim = 16

        # Create untrained VAE
        print("[IsolatedOLA] Creating VAE...")
        self.vae = SimpleVAE(
            input_dim=self.input_dim,
            latent_dim=self.vae_latent_dim,
            hidden_dim=64
        ).to(device)

        # Create PatternLSTM
        print("[IsolatedOLA] Creating PatternLSTM...")
        self.pattern_lstm = PatternLSTM(
            input_dim=self.vae_latent_dim,
            hidden_dim=64,
            output_dim=self.pattern_dim
        ).to(device)
        self.pattern_lstm.reset_state(batch_size=1, device=device)

        # Create GenomeLibrary
        print("[IsolatedOLA] Creating GenomeLibrary...")
        self.genome_library = GenomeLibrary(
            in_dim=self.pattern_dim,
            out_dim=self.genome_out_dim,
            state_dim=128,
            initial_genomes=16,
            max_genomes=32,
            trust_decay=0.990,
            blacklist_threshold=0.68,
            mutation_rate=0.20,
            device=device
        )

        # Create LSH database
        print("[IsolatedOLA] Creating LSH Database...")
        self.lsh_db = LSHGenomeDatabase(
            vector_dim=self.genome_out_dim,
            n_bits=64,
            seed=42
        )

        # Create visualizer
        self.visualizer = None
        if self.visualize:
            print("[IsolatedOLA] Creating Visualizer...")
            self.visualizer = GenomeVisualizer(
                width=1200,
                height=800,
                fps=30
            )

        # State
        self.tick = 0
        self.running = True

        # Performance: cache sys metrics, buffer log writes
        self.update_sys_metrics_every = 50
        self.last_metrics_update_tick = -1
        self.cached_sysvec = torch.zeros(1, 4, device=device)
        self.flush_every = 5000
        self.metrics_buffer = []
        self.genome_metrics_buffer = []
        self.detailed_buffer = []
        self.minimal_buffer = []

        # Minimal metrics + EMA and sampling
        self.consistency_ema = None
        self.ema_alpha = 0.1
        self.cpu_samples = []
        self.ram_samples = []
        self.last_total_mutations_snapshot = 0
        self.last_trust_std = None
        self.last_minimal_metrics = None
        self.health_interval = 200

        # Session directory for metrics logging (sessions/session_0001, 0002, ...)
        base_dir = "sessions"
        os.makedirs(base_dir, exist_ok=True)
        existing = [d for d in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("session_") and d[8:].isdigit()]
        next_num = (max([int(d[8:]) for d in existing]) + 1) if existing else 1
        self.session_dir = os.path.join(base_dir, f"session_{next_num:04d}")
        os.makedirs(self.session_dir, exist_ok=True)

        # Metrics logging setup
        self.metrics_log_path = os.path.join(self.session_dir, "metrics_log.csv")
        with open(self.metrics_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tick", "cpu_usage", "ram_usage", "avg_trust", "avg_consistency"])

        # Per-genome metrics logging setup
        self.genome_metrics_path = os.path.join(self.session_dir, "genome_metrics.csv")
        with open(self.genome_metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "tick",
                "genome_id",
                "trust",
                "consistency",
                "mutation_count",
                "total_ticks"
            ])

        # Feedback loop setup and detailed metrics
        self.feedback_enabled = True
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.current_intensity = 1000

        self.detailed_log_path = os.path.join(self.session_dir, "detailed_metrics.csv")
        with open(self.detailed_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "tick","avg_trust","avg_consistency","cpu_usage","ram_usage",
                "disk_GB_read","temp_C","work_intensity"
            ])

        # Minimal metrics log (every 1k ticks)
        self.minimal_log_path = os.path.join(self.session_dir, "minimal_metrics.csv")
        with open(self.minimal_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "tick","avg_trust","trust_std","avg_consistency_ema",
                "mutations_in_window","nn_similarity_mean","cpu_var","ram_var","healthy"
            ])

        # No external audio input; inputs are generated internally
        self.earn_until = 3000

        print(f"[IsolatedOLA] Initialization complete! Session dir: {self.session_dir}")

    def collect_system_metrics(self):
        # Refresh only every N ticks to reduce overhead; keep last value otherwise
        if self.last_metrics_update_tick == self.tick or (
            self.last_metrics_update_tick >= 0 and (self.tick % self.update_sys_metrics_every) != 0
        ):
            return self.cached_sysvec

        cpu = psutil.cpu_percent(interval=None) / 100.0
        ram = psutil.virtual_memory().percent / 100.0
        # For input vector keep disk/temp at 0.0 to avoid heavy calls per tick
        self.cached_sysvec = torch.tensor([cpu, ram, 0.0, 0.0], device=self.device, dtype=torch.float32).unsqueeze(0)
        # Collect samples for variance calc
        self.cpu_samples.append(float(cpu))
        self.ram_samples.append(float(ram))
        self.last_metrics_update_tick = self.tick
        return self.cached_sysvec

    def generate_random_input(self) -> torch.Tensor:
        """Use live psutil hardware metrics as input"""
        base = self.collect_system_metrics()
        reps = max(1, self.input_dim // 4)
        tiled = base.repeat(1, reps)
        return tiled[:, :self.input_dim]

    def step(self):
        """
        Perform one evolution step
        """
        # 1. Generate random input
        random_input = self.generate_random_input()

        # 2. Encode with VAE
        with torch.no_grad():
            latent = self.vae.sample_latent(random_input)

        # 3. Extract temporal pattern with LSTM
        pattern_vector = self.pattern_lstm.get_pattern(latent)

        # 4. Update all genomes with pattern vector with entropy-gated pruning
        allow_prune = False
        if self.tick % 50 == 0:
            stats_now = self.genome_library.get_library_stats()
            avg_trust_now = float(stats_now['avg_trust'])
            trusts_now = [float(g.stats.trust_score) for g in self.genome_library.genomes]
            trust_std_now = float(np.std(trusts_now)) if trusts_now else 0.0
            genome_ids = [g.genome_id for g in self.genome_library.genomes]
            try:
                entropy = float(self.lsh_db.get_genome_diversity(genome_ids))
            except Exception:
                entropy = 1.0
            # Link pruning to population entropy + trust conditions
            allow_prune = (trust_std_now < 0.03 and avg_trust_now > 0.8 and entropy < 0.4)
        self.genome_library.update_all_genomes(pattern_vector, current_tick=self.tick, allow_prune=allow_prune)

        # Dynamic Decay Modulation: Adjusting the Trust Floor (per-tick)
        stats_now = self.genome_library.get_library_stats()
        avg_trust_now = float(stats_now['avg_trust'])
        current_decay = float(self.genome_library.trust_decay)
        baseline_max = 0.995
        if avg_trust_now >= 0.8:
            # Lower decay (more erosion) when trust is high: 0.992 -> 0.987 as avg_trust goes 0.8 -> 1.0
            t = min(max((avg_trust_now - 0.8) / 0.2, 0.0), 1.0)
            new_decay = 0.992 - 0.005 * t
        else:
            if avg_trust_now < 0.55:
                new_decay = 0.995
            elif avg_trust_now < 0.60:
                new_decay = 0.994
            else:  # 0.60 <= avg_trust_now < 0.80
                new_decay = 0.993
        # Clamp: never raise above baseline_max
        new_decay = min(new_decay, baseline_max)
        if new_decay != current_decay:
            self.genome_library.trust_decay = new_decay
            for g in self.genome_library.genomes:
                g.trust_decay = new_decay
                # Set small positive reinforcement when population trust is very low
                if avg_trust_now < 0.4:
                    g.boost_base_override = 0.002
                else:
                    g.boost_base_override = None

        # 5. Store genome outputs in LSH database (throttled)
        if self.tick % 10 == 0:
            for genome in self.genome_library.genomes:
                with torch.no_grad():
                    output, _ = genome.cell(pattern_vector, genome.h)
                self.lsh_db.store_genome_vector(genome.genome_id, output)

        # Adaptive feedback loop: adjust synthetic CPU workload based on outputs
        if self.feedback_enabled:
            outputs = []
            for g in self.genome_library.genomes:
                if getattr(g, 'output_history', None) and len(g.output_history) > 0:
                    last = g.output_history[-1]
                    try:
                        mag = float(np.mean(np.abs(last)))
                    except Exception:
                        mag = 0.0
                else:
                    mag = 0.0
                outputs.append(mag)
            avg_out = sum(outputs) / len(outputs) if outputs else 0.0
            new_intensity = max(100, min(2000, int(800 + avg_out * 1200)))
            if abs(new_intensity - self.current_intensity) > 250:
                self.current_intensity = new_intensity
                if self.worker_thread and self.worker_thread.is_alive():
                    self.stop_event.set()
                    self.worker_thread.join()
                    self.stop_event.clear()
                self.worker_thread = threading.Thread(
                    target=worker_job, args=(self.stop_event, self.current_intensity), daemon=True
                )
                self.worker_thread.start()

        # 6. Check and mutate low-trust genomes (exploitation cadence outside probation)
        if self.tick >= self.earn_until and (self.tick % 50 == 0):
            # Diversity pressure: small trust penalty for near-duplicates (nearest-neighbor only)
            genome_ids = [g.genome_id for g in self.genome_library.genomes]
            for gid in genome_ids:
                # Find nearest neighbor similarity
                max_sim = 0.0
                for other_id in genome_ids:
                    if other_id == gid:
                        continue
                    sim = self.lsh_db.compute_genome_similarity(gid, other_id)
                    if sim > max_sim:
                        max_sim = sim
                if max_sim > 0.7:
                    g = self.genome_library.get_genome_by_id(gid)
                    if g is not None:
                        g.stats.trust_score = max(0.0, float(g.stats.trust_score) - 0.035)

            # Anti-coast leniency: penalize only if very flat and highly consistent
            flat_eps = 1e-3
            flat_window = 50
            for g in self.genome_library.genomes:
                if getattr(g, 'output_history', None) and len(g.output_history) >= flat_window:
                    # Recent outputs variance
                    try:
                        recent = np.array(g.output_history[-flat_window:])
                        var = float(np.var(recent))
                    except Exception:
                        var = 0.0
                    if g.stats.consistency_score > 0.95 and var < flat_eps:
                        g.stats.trust_score = max(0.0, float(g.stats.trust_score) - 0.01)

            mutated_ids = self.genome_library.check_and_mutate_blacklisted()
            if mutated_ids:
                print(f"[Tick {self.tick}] Mutated genomes: {mutated_ids}")

        # 7. Periodically add new genomes (pause growth while avg_trust < 0.60)
        if self.tick % 200 == 0 and self.tick > 0:
            stats_now = self.genome_library.get_library_stats()
            if stats_now['avg_trust'] >= 0.60:
                new_genome = self.genome_library.add_genome_if_needed()
            else:
                new_genome = None
            if new_genome:
                print(f"[Tick {self.tick}] Added new genome: {new_genome.genome_id}")

        # 8. Update visualization
        if self.visualize and self.visualizer:
            library_stats = self.genome_library.get_library_stats()
            # Update EMA of avg consistency
            if self.consistency_ema is None:
                self.consistency_ema = float(library_stats['avg_consistency'])
            else:
                self.consistency_ema = (
                    self.ema_alpha * float(library_stats['avg_consistency'])
                    + (1.0 - self.ema_alpha) * self.consistency_ema
                )
            self.running = self.visualizer.update(
                self.genome_library.genomes,
                self.lsh_db,
                library_stats,
                self.tick,
                self.last_minimal_metrics
            )

        # 9. Print stats periodically
        if self.tick % 100 == 0:
            self.print_stats()

        # Log hardware/internal/per-genome/detailed metrics every 1000 ticks (buffered)
        if self.tick % 1000 == 0 and self.tick > 0:
            self.log_metrics()
            self.log_genome_metrics()
            stats = self.genome_library.get_library_stats()
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            try:
                disk_counters = psutil.disk_io_counters()
                disk = (disk_counters.read_bytes / 1e9) if disk_counters else 0.0
            except Exception:
                disk = 0.0
            temp_val = 0.0
            try:
                if hasattr(psutil, "sensors_temperatures"):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        first = list(temps.values())[0]
                        if first and len(first) > 0 and getattr(first[0], 'current', None) is not None:
                            temp_val = first[0].current
            except Exception:
                temp_val = 0.0
            self.detailed_buffer.append([
                self.tick,
                round(stats["avg_trust"], 4),
                round(stats["avg_consistency"], 4),
                round(cpu, 2),
                round(ram, 2),
                round(disk, 3),
                round(temp_val, 2),
                self.current_intensity
            ])
            print(f"[Detailed] tick={self.tick} trust={stats['avg_trust']:.3f} cpu={cpu:.1f}% load={self.current_intensity}")

        # Health check and minimal metrics at higher frequency
        if self.tick % self.health_interval == 0 and self.tick > 0:
            stats_h = self.genome_library.get_library_stats()
            trusts = [float(g.stats.trust_score) for g in self.genome_library.genomes]
            trust_std = float(np.std(trusts)) if trusts else 0.0
            avg_trust = float(stats_h['avg_trust'])
            avg_consistency_ema = float(self.consistency_ema if self.consistency_ema is not None else stats_h['avg_consistency'])

            # Mutations in this health window
            total_mut = int(self.genome_library.total_mutations)
            mutations_in_window = total_mut - int(self.last_total_mutations_snapshot)
            self.last_total_mutations_snapshot = total_mut

            # Mean nearest-neighbor similarity
            genome_ids = [g.genome_id for g in self.genome_library.genomes]
            nn_sims = []
            for gid in genome_ids:
                max_sim = 0.0
                for other_id in genome_ids:
                    if other_id == gid:
                        continue
                    sim = self.lsh_db.compute_genome_similarity(gid, other_id)
                    if sim > max_sim:
                        max_sim = sim
                if max_sim > 0.0:
                    nn_sims.append(max_sim)
            nn_similarity_mean = float(sum(nn_sims)/len(nn_sims)) if nn_sims else 0.0

            # Variance of CPU/RAM in last window
            cpu_var = float(np.var(self.cpu_samples)) if self.cpu_samples else 0.0
            ram_var = float(np.var(self.ram_samples)) if self.ram_samples else 0.0
            self.cpu_samples.clear()
            self.ram_samples.clear()

            # Health flag per rule
            N = int(stats_h['total_genomes'])
            K = max(1, N)
            mut_ok = (mutations_in_window >= K and mutations_in_window <= 5*K)
            trust_std_ok = (trust_std >= 0.03 and trust_std <= 0.20)
            healthy = mut_ok and trust_std_ok
            fail_reasons = []
            if not mut_ok:
                fail_reasons.append("mutations out of [N,5N]")
            if not trust_std_ok:
                fail_reasons.append("trust_std outside [0.03,0.20]")
            self.last_trust_std = trust_std

            # Entropy re-injection when converged
            if avg_trust > 0.9 and trust_std < 0.03:
                for g in self.genome_library.genomes:
                    scale = float(np.random.uniform(0.97, 0.99))
                    g.stats.trust_score = max(0.0, min(1.0, float(g.stats.trust_score) * scale))

            # Consistency penalty ceiling: cap per-genome boost via external cap
            boost_cap = max(0.0, 1.0 - avg_consistency_ema)
            for g in self.genome_library.genomes:
                g.external_boost_cap = boost_cap

            self.last_minimal_metrics = {
                'avg_trust': avg_trust,
                'trust_std': trust_std,
                'avg_consistency_ema': avg_consistency_ema,
                'mutations_in_window': mutations_in_window,
                'nn_similarity_mean': nn_similarity_mean,
                'cpu_var': cpu_var,
                'ram_var': ram_var,
                'healthy': healthy,
                'health_reason': None if healthy else "; ".join(fail_reasons) if fail_reasons else "",
            }

            self.minimal_buffer.append([
                self.tick,
                round(avg_trust, 4),
                round(trust_std, 4),
                round(avg_consistency_ema, 4),
                int(mutations_in_window),
                round(nn_similarity_mean, 4),
                round(cpu_var, 6),
                round(ram_var, 6),
                int(1 if healthy else 0)
            ])

        # (Removed) static post-probation decay set; dynamic modulation handles it per-tick

        # Periodically flush buffered logs
        if self.tick % self.flush_every == 0 and self.tick > 0:
            self.flush_logs()

        self.tick += 1

    def print_stats(self):
        """Print current statistics"""
        stats = self.genome_library.get_library_stats()
        lsh_stats = self.lsh_db.get_stats()

        print(f"\n[Tick {self.tick}] Statistics:")
        print(f"  Genomes: {stats['total_genomes']}")
        print(f"  Avg Trust: {stats['avg_trust']:.3f}")
        print(f"  Trust Range: [{stats['min_trust']:.3f}, {stats['max_trust']:.3f}]")
        print(f"  Total Mutations: {stats['total_mutations']}")
        print(f"  Avg Mutation Count: {stats['avg_mutation_count']:.1f}")
        print(f"  Avg Consistency: {stats['avg_consistency']:.3f}")
        print(f"  LSH Vectors: {lsh_stats['total_vectors']}")

        # Top genomes
        top_genomes = self.genome_library.get_top_genomes(k=3)
        print(f"  Top 3 Genomes:")
        for i, genome in enumerate(top_genomes):
            print(f"    {i+1}. {genome}")

    def log_metrics(self):
        """Record hardware metrics and internal stats (buffered)"""
        cpu = psutil.cpu_percent(interval=None) / 100.0
        ram = psutil.virtual_memory().percent / 100.0
        stats = self.genome_library.get_library_stats()
        self.metrics_buffer.append([
            self.tick,
            round(cpu, 4),
            round(ram, 4),
            round(stats["avg_trust"], 4),
            round(stats["avg_consistency"], 4),
        ])
        print(f"[Metrics] tick={self.tick} cpu={cpu:.2f} ram={ram:.2f} trust={stats['avg_trust']:.3f}")

    def log_genome_metrics(self):
        """Save per-genome stats (buffered)."""
        for g in self.genome_library.genomes:
            self.genome_metrics_buffer.append([
                self.tick,
                g.genome_id,
                round(g.stats.trust_score, 4),
                round(g.stats.consistency_score, 4),
                g.stats.mutation_count,
                g.stats.total_ticks
            ])
        print(f"[GenomeMetrics] Logged {len(self.genome_library.genomes)} genomes at tick {self.tick}")

    def flush_logs(self):
        """Flush buffered CSV rows to disk"""
        if self.metrics_buffer:
            with open(self.metrics_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.metrics_buffer)
            self.metrics_buffer.clear()
        if self.genome_metrics_buffer:
            with open(self.genome_metrics_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.genome_metrics_buffer)
            self.genome_metrics_buffer.clear()
        if self.detailed_buffer:
            with open(self.detailed_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.detailed_buffer)
            self.detailed_buffer.clear()
        if self.minimal_buffer:
            with open(self.minimal_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.minimal_buffer)
            self.minimal_buffer.clear()

    def run(self, max_ticks: int = 10000):
        """
        Run the evolution loop

        Args:
            max_ticks: Maximum number of ticks to run (0 = infinite)
        """
        print(f"\n[IsolatedOLA] Starting evolution loop...")
        print(f"[IsolatedOLA] Press ESC or close window to stop\n")

        start_time = time.time()

        try:
            while self.running:
                self.step()

                # Check max ticks
                if max_ticks > 0 and self.tick >= max_ticks:
                    print(f"\n[IsolatedOLA] Reached max ticks ({max_ticks})")
                    break

                # FPS limiting (if no visualizer)
                if not self.visualize:
                    if self.tick % 100 == 0:
                        time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n[IsolatedOLA] Interrupted by user")

        finally:
            # Cleanup
            elapsed = time.time() - start_time
            print(f"\n[IsolatedOLA] Evolution complete!")
            print(f"  Total ticks: {self.tick}")
            print(f"  Elapsed time: {elapsed:.1f}s")
            print(f"  Ticks/sec: {self.tick / max(elapsed, 0.001):.1f}")

            if self.visualizer:
                self.visualizer.close()

            if self.worker_thread and self.worker_thread.is_alive():
                self.stop_event.set()
                self.worker_thread.join()

            # Final stats
            self.print_stats()
            # Ensure logs are flushed
            self.flush_logs()

    def save_checkpoint(self, path: str):
        """
        Save system checkpoint

        Args:
            path: Path to save checkpoint
        """
        self.genome_library.save_checkpoint(path)
        print(f"[IsolatedOLA] Saved checkpoint to {path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Isolated OLA Genome Evolution Visualizer")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to run on")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable visualization (run in terminal only)")
    parser.add_argument("--max-ticks", type=int, default=0,
                        help="Maximum ticks to run (0 = infinite)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to save checkpoint at end")

    args = parser.parse_args()

    # Create system
    system = IsolatedOLA(
        device=args.device,
        visualize=not args.no_viz
    )

    # Run
    system.run(max_ticks=args.max_ticks)

    # Save checkpoint if requested
    if args.checkpoint:
        system.save_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
