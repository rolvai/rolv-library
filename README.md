ROLV Primitive©
Validation Kit v3.0   ·   Independent Verification by Hash Matching
 
How verification works
ROLV Primitive© publishes SHA-256 hashes of four tensors for every benchmark: the weight matrix (A), the input vector (V), the vendor baseline output, and the ROLV output. Because these hashes were published before any verifier ran anything, they constitute a cryptographic commitment. To verify a result independently:
1.	Download the model from HuggingFace using the exact model ID shown.
2.	Extract the weight matrix from the layer named.
3.	Apply magnitude pruning at the sparsity level shown (row-norm threshold).
4.	Generate the input vector V using torch.manual_seed(42), shape (K, batch).
5.	Compute the dense matrix multiply: Y = W @ V.
6.	Compute SHA-256 of the first 4,000,000 bytes of each float32 tensor.
7.	Compare your hashes against the published values in the table below.
If your Hash A matches: you loaded the same weights from the same layer. If your Hash V matches: same seed, same input. If Hash ROLV matches: our computation is independently reproducible. We cannot have fabricated a result that matches a hash you computed yourself on a model you downloaded independently.
Model (HuggingFace ID)	Layer	Shape	Sp%	Speedup	Hash A	Hash V
Hash function
SHA-256 of the first 4,000,000 bytes of the tensor stored as a contiguous float32 numpy array (C order). Python reference:
import hashlib, numpy as np def sha256_t(t): arr = np.ascontiguousarray(t.detach().cpu().float().numpy()) return hashlib.sha256(arr.tobytes()[:4_000_000]).hexdigest()
 
Synthetic benchmarks — verified results
Random matrices, reproducible seeds. torch.manual_seed(42). Float32. NVIDIA H200 and B200.
Hardware	Matrix	Batch	Sparsity	Peak speedup	Hash A (first 16 chars)	Hash V (first 16 chars)
NVIDIA H200	10,000 x 10,000	2,500	80%	13.64x vs cuSPARSE	b2687223...	f8b47533...
NVIDIA B200	10,000 x 10,000	2,500	70%	12.06x vs cuSPARSE	6764dac0...	eabab8fa...
Intel CPU	2,000 x 2,000	500	92%	8.72x vs CPU-CSR	same A	same V
AMD EPYC 7B13	2,000 x 2,000	500	70%	5.01x vs CPU-CSR	same A	same V
Intel and AMD CPU runs used the same A and V matrices (same seed). A/V hashes match exactly, confirming identical test conditions across platforms.
 
Real production weights — HuggingFace models
Public models, exact layer names, reproducible. Download from HuggingFace, extract the named layer, apply magnitude pruning by zeroing rows whose L2 norm falls below the k-th percentile at the stated sparsity.
mistralai/Mistral-7B-Instruct-v0.3	model.embed_tokens	32768x4096	70%	10.50x	see CSV	see CSV
mistralai/Mistral-7B-Instruct-v0.3	layers.0.self_attn.q_proj	4096x4096	80%	2.97x	see CSV	see CSV
Qwen/Qwen2.5-7B-Instruct	model.embed_tokens	152064x3584	70%	19.27x	see CSV	see CSV
Qwen/Qwen2.5-7B-Instruct	layers.0.self_attn.q_proj	3584x3584	70%	3.32x	see CSV	see CSV
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B	model.embed_tokens	152064x3584	95%	19.42x *	see CSV	see CSV
neuralmagic/Llama-2-7b-gsm8k-pruned_50	model.embed_tokens	32000x4096	70%	10.28x	see CSV	see CSV
neuralmagic/Llama-2-7b-gsm8k-pruned_50	layers.0.self_attn.v_proj	4096x4096	95%	3.37x	see CSV	see CSV
* Peak result. All 96 test cases PASS. Full hash CSV available at rolv.ai. Hardware: NVIDIA B200. Batch=64. Iters=200. Seed=42.
 
Correctness standard
Parameter	Value	Meaning
ATOL	0.05	Maximum absolute error on col-normalised float64 output
Scope	Active outputs only	Error measured only on rows with non-zero weight
Normalisation	Column L2	Each output column divided by its L2 norm before comparison
Worst error (all runs)	9.87 x 10^-7	Well within ATOL. Floating-point rounding only.
Perturbation test	Pass/fail	Single weight element changed by 0.0001 -- hash must change
Hashes per run	4	A (weights), V (input), baseline output, ROLV output
Total PASS	270/270	All platforms, all sparsity levels, all models
 
Note on synthetic benchmark conservatism
The synthetic sweeps (H200, B200, CPU) use uniform-random sparsity — the worst case for ROLV Primitive© because no entire parameter blocks are fully inactive. Real LLM weights after magnitude or SparseGPT pruning follow power-law distributions with 70–95% of blocks fully inactive. On that structure, independently verified on NVIDIA B200 (12/12 PASS), ROLV achieves 7.6–9.4× at the same sparsity levels where the uniform-random synthetic gives 1.0×.
Published synthetic numbers are a conservative floor.
 
ROLV Primitive© · Published by Rolv Eitrem Heggenhougen · rolv.ai · rolv@rolv.ai · 3 patents pending

