import numpy as np

def softmax(x):
    
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """
    Q: (batch_size, seq_len_q, d_k)
    K: (batch_size, seq_len_k, d_k)
    V: (batch_size, seq_len_k, d_v)
    """

    d_k = Q.shape[-1]


    scores = np.matmul(Q, K.transpose(0, 2, 1))  # (batch, q_len, k_len)

    scaled_scores = scores / np.sqrt(d_k)


    attention_weights = softmax(scaled_scores)


    output = np.matmul(attention_weights, V)

    return output, attention_weights
    
    


batch_size = 2
seq_len = 4
d_k = 8
d_v = 8

Q = np.random.rand(batch_size, seq_len, d_k)
K = np.random.rand(batch_size, seq_len, d_k)
V = np.random.rand(batch_size, seq_len, d_v)

output, weights = scaled_dot_product_attention(Q, K, V)

print("Output shape:", output.shape)        
print("Weights shape:", weights.shape)      
print(np.sum(weights, axis=-1))  