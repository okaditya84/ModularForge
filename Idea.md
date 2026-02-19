ModularForge: Memory-Bounded Modular Training and Zero-Retraining Assembly of Large Language Models on Resource-Constrained Hardware

1. Problem Statement
   Can we train a large language model (e.g., 1B parameters) entirely on a low-end consumer device (8GB RAM, no GPU) by decomposing it into many small, independently trainable modules (e.g., 100 × 10M parameters), each trained sequentially on different data subsets, and then assemble the final model through a streaming, memory-bounded process — without any retraining, fine-tuning, or ever loading more than one module into memory at any stage?

This problem sits at the unexplored intersection of three active research areas:

Modular NeuralNetwork Training
ModularForge(THIS WORK)
Model Merging& Fusion
Memory-EfficientML on Edge
Train Big Modelson Small Devices
The Core Challenge (In Simple Terms)
Imagine you have a laptop with only 8GB of RAM. You want to build a 1-billion-parameter language model. Current approaches require you to either:

Buy expensive hardware — Train the full 1B model on cloud GPUs ($10,000+)
Use parameter-efficient methods (LoRA, QLoRA) — These adapt an existing model, they don't build one from scratch
Use model parallelism — Requires multiple GPUs communicating simultaneously
Our question is radical: What if you could train 10M parameters at a time, sequentially, on different slices of data — then stitch all the pieces together at the end, never exceeding 10M parameters in memory at any point?

IMPORTANT

The key insight: Not just modular training, but memory-bounded assembly. Even if you train modules independently, every existing fusion method (mergekit, FuseLLM, task arithmetic) requires loading the full target model into memory for the merge step. Our problem demands that assembly itself be a streaming, constant-memory operation.

2. Why This Is Novel — What Exists and Why It's Not Enough
   2.1 Existing Model Merging Methods
   Method	What It Does	Why It Doesn't Solve Our Problem
   Task Arithmetic (ICLR 2023)	Combines "task vectors" (weight deltas from fine-tuned models) via addition/subtraction	Requires a shared base model; all models must have identical architecture; merging loads full model
   TIES-Merging (NeurIPS 2023)	Adds sparsification and sign consensus to task arithmetic to reduce interference	Same as above — fine-tuned variants from ONE base model, not independently trained modules
   DARE (2024)	Random pruning + rescaling of task vectors for robust merging	Same shared-base assumption; requires identical architecture
   FuseLLM (ICLR 2024)	Fuses probabilistic distributions from different LLMs via lightweight continual training	Requires 0.1% of original training tokens for fine-tuning; needs all source models loaded for probability extraction
   Mergekit (Arcee AI)	Toolkit implementing Linear, SLERP, TIES, DARE, passthrough merging	Requires identical architectures; lazy loading helps but still loads full merged model structure
   Locate-then-Merge (EMNLP 2025)	Neuron-level parameter fusion for multimodal LLMs	Designed for merging fine-tuned vs. base model; requires neuron significance analysis across full model
   CAUTION

Critical gap in ALL existing merging methods: They merge models that were fine-tuned from a common base model. None of them build a model from scratch by combining independently trained parts that never shared weights, gradients, or architecture initialization.

2.2 Existing Modular Training Approaches
Method	What It Does	Why It Doesn't Solve Our Problem
BTX (Branch-Train-MiX) (2024)	Branches from a seed LLM, independently trains copies on domain-specific data, combines FFN weights into MoE	Starts from a pre-trained seed model (not from scratch); router requires joint fine-tuning on combined data
BIMT (2023)	Brain-inspired training embedding neurons in geometric space for modular structure	Focuses on interpretability, not assembly; trains one network end-to-end, not independent modules
Progressive Training (arXiv 2025)	Grows 1B→2B→4B→8B by expanding smaller models	Requires loading full model at each stage; doesn't independently train separate components
LoRA-LEGO (2024)	Disassembles/reassembles LoRA adapters at "Minimal Semantic Unit" granularity	Requires a pre-trained base model; LoRA adapters are tiny relative to full model
2.3 Memory-Efficient Training Techniques
Method	What It Does	Why It Doesn't Solve Our Problem
ZeRO-Infinity (DeepSpeed)	Offloads optimizer states/gradients to NVMe	Still requires the model to exist as one entity; handles training one model across tiers
Gradient Checkpointing	Trades compute for memory by recomputing activations	Reduces memory during training but still needs full model loaded
lazy-transformers-merge	Merges models using lazy tensors without loading all into RAM	Only does weighted averaging of architecturally identical models; doesn't handle independently trained heterogeneous modules
Mediator (Feb 2025)	Layer-wise averaging + task-level expert routing with CPU offloading	Designed for fine-tuned model variants; still requires significant memory for expert routing decisions
Quantization (QLoRA, GPTQ)	Reduces precision to fit models in memory	Adapts/compresses existing models, doesn't train from scratch
2.4 Federated Learning Connection
Federated learning trains models across distributed clients and aggregates them, which is conceptually related. However:

FedAvg and variants assume identical model architectures across all clients
Aggregation happens at the gradient level during training rounds, not post-hoc assembly
Heterogeneous federated learning (HFedCMF, FedMRL) uses knowledge distillation — which requires additional data and compute
None address the setting where each "client" trains a different portion of the final model
3. The Specific Novel Contributions Required
Our problem demands solving four sub-problems that are currently unsolved together:

Sub-Problem 1: Modular Architecture Design
Design a transformer variant where the full model can be cleanly decomposed into N independent modules, each trainable without any cross-module parameter sharing, gradient flow, or communication.

Possible directions:

Layer-wise decomposition: Each module = a contiguous block of transformer layers. Modules are trained with a shared (frozen) embedding layer and a local prediction head
MoE-style expert decomposition: Each module = an expert FFN. At assembly, a lightweight router is concatenated (router weights can be random-initialized or heuristic-based)
Attention-head partitioning: Each module = a subset of attention heads operating on the same embedding space
Hybrid approach: Combine layer blocks with parallel expert branches within blocks
Sub-Problem 2: Data-Aware Training Protocol
How to partition the training corpus across N modules such that each module learns complementary, non-redundant knowledge that remains useful when composed.

Key insight: This is fundamentally different from data parallelism. In data parallelism, each device sees different batches but trains the same model. Here, each module is a different model seeing different data.

Possible directions:

Topic/domain-based partitioning: Module 1 = code, Module 2 = math, Module 3 = science, etc.
Curriculum-based partitioning: Module i trained on data of increasing complexity level i
Random partitioning with overlap: Each module sees a random subset with ~10% overlap for coherence
Embedding-based clustering: Pre-cluster all training data by sentence embeddings, assign clusters to modules
Sub-Problem 3: Memory-Bounded Streaming Assembly
Assemble N trained modules (each ≤ M parameters) into a final model (N × M parameters) using O(M) memory throughout the assembly process. Never load more than one module at a time.

This is the most novel technical contribution. The algorithm:

python

# Pseudocode: Streaming Assembly with O(M) memory

def streaming_assembly(module_dir, output_path, N, M):
    """
    Assemble N modules into final model using O(M) memory.
    Never loads more than one module at a time.
    """
    # Step 1: Write model metadata/config (few KB)
    write_model_config(output_path, total_params=N*M)

    # Step 2: Initialize output file with correct structure
    # Using safetensors format with pre-allocated offsets
    init_empty_safetensors(output_path, total_size=N*M*4)  # 4 bytes per param (fp32)

    # Step 3: Stream each module into its designated position
    for i in range(N):
        module = load_module(f"{module_dir}/module_{i}.safetensors")  # O(M) memory

    # Memory-mapped write to specific byte offset
        write_params_at_offset(
            target=output_path,
            params=module.state_dict(),
            byte_offset=compute_offset(i, M)
        )

    del module  # Free memory immediately
        gc.collect()

    # Step 4: (Optional) Write alignment/routing metadata
    # For MoE: generate router weights heuristically
    write_router_metadata(output_path, N)

    # Total peak memory: O(M) + O(metadata) ≈ O(M)
Key technical challenges:

Safetensors/GGUF format allows random-access writes to specific tensor positions
Must handle tokenizer, embedding layers, and layer norms shared across modules
Need a convention for parameter naming that maps module-local names to global positions
Sub-Problem 4: Coherence Without Retraining
The assembled model must produce coherent, useful outputs without any fine-tuning, alignment, or additional training after assembly.

This is the hardest open question. Two main strategies:

Strategy A: Shared Embedding Space

All modules share the same (pre-computed or frozen) embedding layer
Modules are trained to operate in a consistent latent space
Assembly preserves the embedding-to-layer mapping
Strategy B: Modular Interface Contracts

Define fixed-dimensionality interfaces between modules
Each module's input/output is a standardized tensor format
Modules are trained independently but with compatible interface constraints
Strategy C: Post-Assembly Calibration (Minimal retraining)

Allow a tiny calibration step (~0.01% of training data) that runs on the assembled model
This calibration only adjusts layer norms and biases (not full weights)
Keeps memory requirement much smaller than full model (norms/biases are ≪ 1% of parameters)
4. What Makes This Problem Solvable (Feasibility Arguments)
4.1 Evidence from Existing Research
BTX (Branch-Train-MiX) shows that independently trained model copies can be recombined into a working MoE — but starting from a shared base. Our work removes the shared-base assumption.

NeurIPS 2024 Merging Competition demonstrated that DARE-TIES can effectively merge models without retraining — but all from Llama-3-8B-Instruct variants. We extend this to independently initialized modules.

lazy-transformers-merge proves that O(M) memory assembly is technically feasible for the file-writing step — but only for identical architectures doing weighted averaging.

Modular neural network research (2021 Neural Networks paper) provides guidelines for designing independently trainable modules with composable interfaces.

MoE architectures naturally partition knowledge across experts, and routers can be extremely lightweight (even random routing produces non-trivial results, per early MoE literature).

4.2 Why MoE Architecture Is the Most Promising Path
The Mixture of Experts architecture is the best-fit for this problem because:

Assembly Phase (Streaming, O(M) memory)
Training Phase (Sequential, O(M) memory)
Module 110M paramsCode data
Module 210M paramsMath data
Module N10M paramsGeneral data
Shared Layers(Embedding, LN)
Expert 1
Expert 2
Expert N
Router(Heuristic)
Experts are naturally independent — each FFN block processes tokens independently
The router can be initialized heuristically (uniform, random, or based on training data statistics)
Shared components (embedding, layer norms) are small relative to expert weights
At inference, only a subset of experts are active (sparse gating), so memory during inference is also reduced
4.3 Concrete Feasibility Estimate
For a 1B parameter model assembled from 100 × 10M modules:

Component	Memory During Training	Memory During Assembly	Memory During Inference
One module (10M params, fp32)	~40 MB	~40 MB	—
Optimizer states (Adam, 2x)	~80 MB	—	—
Gradients	~40 MB	—	—
Full model on disk	—	—	~4 GB (fp32), ~1 GB (int8)
Peak RAM needed	~160 MB	~40 MB	~1 GB (quantized)
TIP

With int8 quantization during inference, even the assembled 1B model can run on 8GB RAM. The training and assembly phases need only ~160 MB peak — achievable on virtually any modern device.

5. Proposed Research Framework: "ModularForge"
   5.1 Framework Components
   Component	Description	Status
   ModularForge-Arch	Modular transformer architecture with independent expert FFN blocks, shared embeddings, and a lightweight router	Novel design needed
   ModularForge-Train	Sequential training pipeline that trains one module at a time on a specific data partition, with frozen shared components	Buildable with PyTorch
   ModularForge-Assemble	Streaming O(M) memory assembly algorithm using memory-mapped files and safetensors format	Novel algorithm needed
   ModularForge-Evaluate	Benchmark suite comparing assembled model against monolithic baseline on perplexity, downstream tasks, and resource usage	Standard benchmarking
   5.2 Experimental Plan
   Phase 1: Proof of Concept (Small Scale)

Architecture: 50M parameter model from 5 × 10M modules
Data: WikiText-103 (100M tokens), partitioned 5 ways
Baseline: Monolithically trained 50M model
Metrics: Perplexity, downstream task accuracy, assembly time, peak memory
Phase 2: Scaling Study

Architecture: 500M parameter model from 50 × 10M modules
Data: Pile subset (1B tokens), partitioned 50 ways
Compare: MonolithicVs. ModularForge vs. BTX (with/without shared base)
Phase 3: Full-Scale Demonstration

Architecture: 1B+ parameter model from 100+ × 10M modules
Data: Full training corpus (10B+ tokens)
Demonstrate: End-to-end training on a single 8GB RAM device
5.3 Key Research Questions to Answer
How does data partitioning strategy affect final model quality? (random vs. topic-based vs. curriculum)
What is the minimum overlap needed between module training data for coherent assembly?
Can a heuristically initialized router achieve >90% of fine-tuned router performance?
How does the number of modules N affect the quality-efficiency tradeoff? (few large modules vs. many small modules)
Does the order of sequential training matter? (easiest data first vs. random order)
Can post-assembly norm calibration (O(1%) memory) significantly boost quality?
6. Related Work — Complete Reference List
Papers (chronological)
Year	Paper	Venue	Key Contribution	Relation to Our Work
2021	Design and independent training of composable neural modules	Neural Networks	Guidelines for modular interfaces	Theoretical foundation for our interface design
2023	Editing Models with Task Arithmetic	ICLR	Task vectors for model editing via weight arithmetic	We go beyond: no shared base model
2023	TIES-Merging: Resolving Interference When Merging Models	NeurIPS	Sparsification + sign consensus for many-model merging	Reduces interference, but assumes shared-base fine-tuning
2023	BIMT: Brain-Inspired Modular Training	arXiv:2305.08746	Geometric embedding for modular, interpretable networks	Related architecture idea, but end-to-end training
2024	FuseLLM: Knowledge Fusion of LLMs	ICLR	Probabilistic distribution fusion across architectures	Needs 0.1% retraining; loads all source models
2024	DARE: Drop And REscale	—	Random pruning + rescaling for robust task vector merging	Additional merging technique, same limitations as Task Arithmetic
2024	BTX: Branch-Train-MiX	—	Branch from seed LLM, independently train, combine as MoE	Closest work, but requires pre-trained seed model
2024	Model Merging in LLMs, MLLMs, and Beyond (Survey)	arXiv:2408.07666	Comprehensive survey of model merging	Documents the field landscape and open problems
2024	NeurIPS LLM Merging Competition	NeurIPS	Community benchmark for model merging	Winning method: DARE-TIES on Llama-3 variants
2025	Progressive Training Using Model Expansion	arXiv:2504.00623	Grow 1B→8B progressively, save 25% compute	Requires full model at each stage
2025	Locate-then-Merge: Neuron-Level Fusion	EMNLP	Neuron-level parameter merging for multimodal LLMs	Preserves visual capabilities, but for fine-tuned pairs
2025	Mediator: Memory-efficient LLM Merging	arXiv	Layer-wise averaging + expert routing with reduced memory	Closest to memory efficiency goal, but merges fine-tuned variants
2025	AIMMerging: Adaptive Iterative Model Merging	EMNLP	Dynamic merging based on training trajectory monitoring	Temporal merging, but for continual learning
Open-Source Tools
Tool	Stars	Key Feature	Limitation for Our Problem
mergekit	5.8k+	YAML-configurable merging with lazy loading	Requires identical architectures; merges existing models
lazy-transformers-merge	Small	O(M) memory merging via lazy tensors	Only weighted averaging
FuseLLM	—	Cross-architecture knowledge fusion	Requires lightweight retraining
7. Why This Is Publication-Worthy
7.1 Novelty Checklist
Criterion	Status	Justification
Novel problem formulation	✅ Yes	No prior work formulates the bounded-memory, from-scratch modular training + streaming assembly problem
Practical motivation	✅ Yes	Democratizes LLM training for researchers without GPU access
Theoretical contribution	✅ Yes	Memory complexity analysis, convergence guarantees for modular composition
Open-source framework	✅ Yes	Reusable toolkit for community
Experimental validation	✅ Yes	Clear baselines and scaling study
7.2 Target Venues
Tier 1: NeurIPS, ICML, ICLR (main conference)
Workshops: NeurIPS Efficient ML Workshop, ICML Workshop on Federated Learning
Journals: JMLR, TMLR
Preprint: arXiv (cs.LG, cs.CL)
7.3 Impact
Democratization of AI: Anyone with a laptop can train a custom LLM
Green AI: Reduces energy consumption by avoiding redundant full-model training
Privacy: Different modules can be trained on different private datasets without data sharing
Collaborative AI: Open-source community can contribute individual modules to a collective model
Edge AI: Enables model construction directly on edge devices
8. Recommended Next Steps
Immediate Actions (Week 1-2)
Set up the project repo — Create ModularForge on GitHub with proper README, license (Apache 2.0), and structure
Implement the modular architecture — Start with a simplified MoE transformer (50M params) in PyTorch
Implement the streaming assembler — Build the O(M) memory assembly tool using safetensors
Short-Term (Week 3-6)
Run proof-of-concept experiments — Train 5 × 10M modules on WikiText-103 partitions
Evaluate against baselines — Compare perplexity vs. monolithic training
Iterate on data partitioning — Test random vs. topic-based vs. curriculum strategies
Medium-Term (Week 7-12)
Scale up — Attempt 500M and 1B parameter assemblies
Write the paper draft — Introduction, related work, method, experiments, analysis
Community feedback — Post to arXiv, share on Reddit, Twitter, HuggingFace
9. One-Line Summary
ModularForge is a framework for training large language models on consumer hardware by decomposing the model into small, independently trainable modules and assembling them through a streaming, memory-bounded process that never exceeds the memory footprint of a single module — addressing the unsolved intersection of modular training, model merging, and resource-constrained ML.
