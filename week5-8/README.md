\#  Understanding the Transformer Architecture (From Intuition to Code)



\##  Why Transformers Matter



Modern AI systems like \*\*BERT\*\* and \*\*GPT\*\* are built on one core idea:



> Instead of processing text sequentially, process everything at once using attention.



This shift is what made Large Language Models (LLMs) powerful.



\---



\##  The Big Picture: Transformer Architecture



!\[Transformer Architecture](https://jalammar.github.io/images/t/transformer\_resideual\_layer\_norm\_3.png)



A Transformer consists of:



\### Encoder

\- Reads the input sentence

\- Builds contextual representations



\### Decoder

\- Generates output (used in GPT-style models)



Each layer contains:

1\. Self-Attention

2\. Feedforward Network

3\. Residual connections + LayerNorm



\---



\##  Core Idea: Self-Attention



Self-attention lets each word ask:



> “Which other words in this sentence matter to me?”



\---



\##  The Attention Formula



Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V





Where:

\- \*\*Q (Query)\*\* → what the token is looking for  

\- \*\*K (Key)\*\* → what other tokens offer  

\- \*\*V (Value)\*\* → actual information  



\---



\##  How It Works (Step-by-Step)



!\[Attention Mechanism](https://jalammar.github.io/images/t/self-attention-matrix-calculation.png)



\### Step 1 — Similarity



Q × Kᵀ





\### Step 2 — Scaling



÷ √dₖ





\### Step 3 — Softmax

Convert scores into probabilities



\### Step 4 — Weighted Sum

Combine values using attention weights



\---



\##  Code: Scaled Dot-Product Attention (From Scratch)



```python

import torch

import torch.nn.functional as F



def scaled\_dot\_product\_attention(Q, K, V, mask=None):

   d\_k = Q.size(-1)

   

   scores = torch.matmul(Q, K.transpose(-2, -1))

   scores = scores / torch.sqrt(torch.tensor(d\_k, dtype=torch.float32))



   if mask is not None:
   scores = scores.masked\_fill(mask == 0, float('-inf'))



   attention\_weights = F.softmax(scores, dim=-1)

   output = torch.matmul(attention\_weights, V)



   return output, attention\_weights


```






\## Why Scaling Matters



Without scaling:



Dot products become large

Softmax saturates

Gradients vanish



\## Scaling stabilizes training



\## Multi-Head Attention



Instead of one attention operation, Transformers use multiple heads.



Each head learns:



Syntax (grammar)

Coreference (“it” → “cat”)

Long-range dependencies

\## Positional Encoding



(https://jalammar.github.io/images/t/transformer\_positional\_encoding\_vectors.png)



Transformers don’t know word order by default.



Solution: add positional encodings using sine/cosine functions.



\## Visualizing Attention



We used bertviz to explore attention patterns.



Example Sentence

The cat sat on the mat because it was tired.

\## Observations

1\. Syntax

“the” → “cat”

“cat” → “sat”

2\. Coreference

“it” → “cat”

3\. Local Attention

Words attend to nearby tokens

\##🔬 Attention Heatmap



Diagonal → local context

Off-diagonal → long-range dependencies

Strong columns → important tokens

