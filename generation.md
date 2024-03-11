Generation Size explosion
---
When creating a dataset, using FLAN, there are 2 main factors controlling generated dataset size:
* The number of sampled documents, which refers to the number of chunks used for generation. denote the number of sampes with  $n_s$
* percentiles, which is fied to : [0.05, 0.25, 0.5, 0.95] denote the number odf the number of percentiles with  $n_p$

Then the generated dataset will be the size of:
$$ n_s * n_p * 3 $$
because there are two ways to generate contradictionary examples