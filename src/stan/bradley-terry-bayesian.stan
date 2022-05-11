data {
  int<lower=0> K; // agents
  int<lower=0> N; // comparisons
  array[N] int<lower=1, upper=K> i;  // agent i for comparison n
  array[N] int<lower=1, upper=K> j;  // agent j for comparison n
  array[N] int<lower=0, upper=1> y;  // winner for comparison n
}
parameters {
  real alpha_0;     // home-court advantage
  vector[K] alpha;  // ability for agent
}
model {
  alpha_0 ~ normal(0, 1);
  alpha ~ normal(0, 1);
  y ~ bernoulli_logit(alpha_0 + alpha[i] - alpha[j]);
}
generated quantities {
  array[K] int<lower=1, upper=K> ranking; // rank of player ability
  {
    array[K] int ranked_index = sort_indices_desc(alpha);
    for (k in 1 : K) {
      ranking[ranked_index[k]] = k;
    }
  }
  vector[K] probability = softmax(alpha); // probability
}