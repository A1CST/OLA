# Isolated OLA - Genome Evolution Visualizer

A self-contained evolution visualization system that shows how genome structure and trust change over time through continuous mutation.

## Overview

This project visualizes real-time evolution dynamics of neural genomes without any game or environment dependencies. It demonstrates:

- **Continuous Evolution**: Genomes mutate based on trust scores
- **Trust Dynamics**: Trust increases with output consistency, decays over time
- **Real-time Visualization**: Live network graph showing genome relationships
- **LSH Similarity**: Genomes connected by similarity edges

## System Architecture

```
Random Input → VAE → PatternLSTM → Genomes → Mutation → Visualization
                ↓         ↓           ↓          ↓
            Latent    Pattern    Outputs     Trust
            Vector   Features   +Trust      Update
```

### Components

1. **VAE** (`vae.py`): Untrained variational autoencoder that encodes random inputs into latent vectors
2. **PatternLSTM** (`pattern_lstm.py`): Temporal model that extracts patterns from latent sequences
3. **OLAGenome** (`ola_genome.py`): Evolvable neural network with trust tracking and mutation capabilities
4. **GenomeLibrary** (`genome_library.py`): Manages population of genomes with automatic mutation
5. **LSHGenomeDatabase** (`lsh_genome_database.py`): Stores genome outputs and computes similarities
6. **Visualizer** (`visualizer.py`): Real-time pygame visualization of genome network

## Visualization Features

### Node Properties
- **Color**: Trust score (Blue = low trust, Red = high trust)
- **Size**: Number of mutations undergone
- **Label**: Genome ID

### Edge Properties
- **Connection**: LSH similarity between genomes
- **Thickness**: Similarity strength
- **Visibility**: Only shown above threshold (0.6)

## Installation

```bash
cd isolated_OLA
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy 1.24+
- Pygame 2.5+

## Usage

### Basic Usage

```bash
python main.py
```

### Command Line Options

```bash
python main.py --device cuda              # Run on GPU
python main.py --no-viz                   # Disable visualization (terminal only)
python main.py --max-ticks 5000           # Run for 5000 ticks then stop
python main.py --checkpoint save.pt       # Save checkpoint at end
```

### Controls

- **ESC**: Exit the visualization
- **Close Window**: Stop the evolution loop

## How It Works

### Evolution Loop

1. **Input Generation**: Random smooth signals using sine waves
2. **Encoding**: VAE encodes input to latent space
3. **Pattern Extraction**: PatternLSTM processes latent sequence
4. **Genome Update**: All genomes process pattern vector
5. **Trust Update**: Trust increases with consistency, decays over time
6. **Mutation Check**: Every 50 ticks, low-trust genomes mutate
7. **Population Growth**: New genomes added every 200 ticks
8. **Visualization**: Network graph updates at 30 FPS

### Trust Mechanics

- **Decay**: Trust multiplied by 0.995 every tick
- **Boost**: Consistent outputs increase trust
- **Consistency**: Measured by variance across recent outputs
- **Mutation Threshold**: Genomes below 0.3 trust are mutated

### Mutation Strategy

- **Adaptive Rate**: Higher mutation rate for lower trust
- **Growth Probability**: Chance to expand network capacity
- **Multiple Passes**: More mutations for very low trust
- **Trust Reset**: After mutation, trust reset to 0.8

## Project Structure

```
isolated_OLA/
├── main.py                    # Main evolution loop
├── vae.py                     # Untrained VAE encoder/decoder
├── pattern_lstm.py            # Temporal pattern extraction
├── ola_genome.py              # Evolvable genome with trust
├── genome_library.py          # Population management
├── lsh_genome_database.py     # Similarity tracking
├── visualizer.py              # Real-time visualization
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Key Parameters

### Genome Configuration
- `in_dim`: 32 (pattern vector dimension)
- `out_dim`: 16 (genome output dimension)
- `state_dim`: 128 (recurrent state size)
- `trust_decay`: 0.995 (per-tick decay factor)

### Population Settings
- `initial_genomes`: 12 (starting population)
- `max_genomes`: 32 (population cap)
- `blacklist_threshold`: 0.3 (mutation trigger)
- `mutation_rate`: 0.15 (base mutation probability)

### Evolution Dynamics
- Mutation check: Every 50 ticks
- New genome addition: Every 200 ticks
- Stats printing: Every 100 ticks

## Example Output

```
[Tick 500] Statistics:
  Genomes: 15
  Avg Trust: 0.512
  Trust Range: [0.245, 0.782]
  Total Mutations: 23
  Avg Mutation Count: 1.5
  Avg Consistency: 0.634
  LSH Vectors: 150
  Top 3 Genomes:
    1. OLAGenome(id=3, trust=0.782, ticks=500, mutations=1, consistency=0.823)
    2. OLAGenome(id=7, trust=0.691, ticks=500, mutations=2, consistency=0.745)
    3. OLAGenome(id=1, trust=0.623, ticks=500, mutations=3, consistency=0.612)
```

## Performance

- **CPU**: ~100-200 ticks/sec with visualization
- **GPU**: ~150-300 ticks/sec with visualization
- **No Viz**: ~500-1000 ticks/sec (terminal only)

## Future Enhancements

- Interactive mutation controls
- Genome ancestry tracking
- Diversity metrics visualization
- Checkpoint loading/resuming
- Custom input patterns
- Network topology evolution
- Multi-population competition

## License

This is a research and educational project. Feel free to modify and extend.
