# Sanity / randomization — multi-case, multi-seed suite

Similarity of the randomized-model attribution to the trained-model attribution. A method passes when similarity DECAYS to the spatial-baseline level as the network is randomized. `collapse` = fraction of runs whose map went (near-)constant — correlations are NaN there by construction and `energy_ratio`=std_rand/std_ref→0 is the collapse evidence.


## 1. Cascading layer randomization (output→input, cumulative)

Aggregated over seeds × cases. Stage 0 = only the Perceiver decoder randomized; last stage = whole network randomized.


### ig
| stage | params | pearson | spearman | cosine | ssim | top1% | energy_ratio | collapse |
|--|--:|--:|--:|--:|--:|--:|--:|--:|
| 0:perceiver_decoder | 0.046±0.000 | 0.012±0.091 | -0.073±0.193 | 0.012±0.091 | 0.999±0.001 | 0.557±0.037 | 0.007±0.006 | 0% |
| 1:backbone_decoder_layers | 0.490±0.000 | -0.160±0.180 | -0.007±0.035 | -0.160±0.180 | 0.999±0.001 | 0.574±0.027 | 0.003±0.002 | 33% |
| 2:backbone_mid | 0.496±0.000 | -0.023±0.154 | 0.004±0.033 | -0.023±0.154 | 0.999±0.001 | 0.574±0.026 | 0.002±0.002 | 33% |
| 3:backbone_encoder_layers | 0.936±0.000 | -0.100±0.105 | 0.000±0.005 | -0.100±0.105 | 0.999±0.001 | 0.545±0.025 | 0.004±0.003 | 33% |
| 4:perceiver_encoder | 1.000±0.000 | — | — | — | — | 0.543±0.027 | 0.000±0.000 | 100% |

### rise
| stage | params | pearson | spearman | cosine | ssim | top1% | energy_ratio | collapse |
|--|--:|--:|--:|--:|--:|--:|--:|--:|
| 0:perceiver_decoder | 0.046±0.000 | 0.411±0.770 | 0.402±0.761 | 0.411±0.770 | 0.104±0.004 | 0.659±0.192 | 0.003±0.002 | 33% |
| 1:backbone_decoder_layers | 0.490±0.000 | -0.108±0.863 | -0.114±0.850 | -0.109±0.863 | 0.102±0.003 | 0.497±0.318 | 0.001±0.000 | 67% |
| 2:backbone_mid | 0.496±0.000 | 0.858±0.113 | 0.846±0.118 | 0.857±0.113 | 0.104±0.001 | 0.438±0.284 | 0.001±0.001 | 67% |
| 3:backbone_encoder_layers | 0.936±0.000 | 0.829±0.152 | 0.817±0.162 | 0.828±0.153 | 0.105±0.003 | 0.423±0.226 | 0.001±0.002 | 67% |
| 4:perceiver_encoder | 1.000±0.000 | — | — | — | — | 0.139±0.162 | 0.000±0.000 | 100% |

### saliency
| stage | params | pearson | spearman | cosine | ssim | top1% | energy_ratio | collapse |
|--|--:|--:|--:|--:|--:|--:|--:|--:|
| 0:perceiver_decoder | 0.046±0.000 | 0.023±0.157 | -0.068±0.187 | 0.023±0.157 | 0.992±0.006 | 0.581±0.032 | 0.103±0.115 | 0% |
| 1:backbone_decoder_layers | 0.490±0.000 | 0.007±0.049 | 0.012±0.041 | 0.007±0.049 | 0.992±0.006 | 0.554±0.058 | 0.031±0.036 | 0% |
| 2:backbone_mid | 0.496±0.000 | -0.007±0.050 | -0.006±0.028 | -0.007±0.050 | 0.992±0.006 | 0.549±0.066 | 0.020±0.021 | 0% |
| 3:backbone_encoder_layers | 0.936±0.000 | 0.032±0.032 | -0.003±0.009 | 0.032±0.032 | 0.992±0.006 | 0.490±0.124 | 0.031±0.034 | 0% |
| 4:perceiver_encoder | 1.000±0.000 | 0.027±0.008 | 0.001±0.002 | 0.027±0.008 | 0.995±0.000 | 0.479±0.129 | 0.001±0.001 | 67% |

### vit_cx
| stage | params | pearson | spearman | cosine | ssim | top1% | energy_ratio | collapse |
|--|--:|--:|--:|--:|--:|--:|--:|--:|
| 0:perceiver_decoder | 0.046±0.000 | 0.490±0.777 | 0.080±0.117 | 0.496±0.755 | 0.934±0.054 | 0.645±0.111 | 0.004±0.003 | 33% |
| 1:backbone_decoder_layers | 0.490±0.000 | -0.159±0.584 | 0.035±0.042 | -0.134±0.594 | 0.916±0.051 | 0.608±0.242 | 0.001±0.000 | 50% |
| 2:backbone_mid | 0.496±0.000 | 0.380±0.185 | 0.035±0.007 | 0.414±0.161 | 0.880±0.000 | 0.362±0.222 | 0.001±0.001 | 67% |
| 3:backbone_encoder_layers | 0.936±0.000 | 0.132 | 0.031 | 0.124 | 0.880 | 0.244±0.212 | 0.001±0.002 | 83% |
| 4:perceiver_encoder | 1.000±0.000 | — | — | — | — | 0.013±0.011 | 0.000±0.000 | 100% |

## 2. Full randomization — all params, multiple seeds (mean±std over seeds×cases)

| method | pearson | spearman | cosine | ssim | top1% | energy_ratio | collapse |
|--|--:|--:|--:|--:|--:|--:|--:|
| ig | — | — | — | — | 0.543±0.025 | 0.000±0.000 | 100% |
| rise | — | — | — | — | 0.168±0.179 | 0.000±0.000 | 100% |
| saliency | 0.009±0.021 | -0.000±0.002 | 0.009±0.021 | 0.995±0.000 | 0.478±0.128 | 0.001±0.001 | 67% |
| vit_cx | — | — | — | — | 0.044±0.075 | 0.000±0.000 | 100% |

## 3. Spatial-baseline calibration (trained attribution vs naive maps)

What a similarity value *means*: a randomized-model score at or below the `iid_noise` row is genuine collapse.

| case | method | baseline | pearson | spearman | cosine | ssim | top1% |
|--|--|--|--:|--:|--:|--:|--:|
| alps_w | ig | gaussian_blob | -0.040 | 0.006 | -0.040 | 0.957 | 0.446 |
| alps_w | ig | iid_noise | 0.000 | 0.000 | 0.000 | -0.000 | 0.010 |
| alps_w | ig | value_shuffled | 0.000 | -0.000 | 0.000 | 0.997 | 0.011 |
| alps_w | saliency | gaussian_blob | 0.002 | -0.009 | 0.002 | 0.984 | 0.184 |
| alps_w | saliency | iid_noise | -0.000 | -0.000 | -0.000 | 0.000 | 0.010 |
| alps_w | saliency | value_shuffled | 0.000 | 0.000 | 0.000 | 0.983 | 0.010 |
| atl_s | ig | gaussian_blob | 0.055 | -0.000 | 0.055 | 0.970 | 0.487 |
| atl_s | ig | iid_noise | -0.000 | -0.000 | -0.000 | 0.000 | 0.010 |
| atl_s | ig | value_shuffled | -0.000 | 0.000 | -0.000 | 0.983 | 0.010 |
| atl_s | saliency | gaussian_blob | 0.063 | 0.004 | 0.063 | 0.970 | 0.612 |
| atl_s | saliency | iid_noise | -0.000 | 0.001 | -0.000 | -0.000 | 0.010 |
| atl_s | saliency | value_shuffled | 0.000 | 0.001 | 0.000 | 0.976 | 0.011 |
| trop_s | ig | gaussian_blob | -0.079 | -0.007 | -0.079 | 0.956 | 0.496 |
| trop_s | ig | iid_noise | 0.000 | 0.000 | 0.000 | 0.000 | 0.011 |
| trop_s | ig | value_shuffled | -0.000 | 0.000 | -0.000 | 0.990 | 0.010 |
| trop_s | saliency | gaussian_blob | -0.017 | 0.035 | -0.017 | 0.976 | 0.154 |
| trop_s | saliency | iid_noise | -0.000 | 0.000 | -0.000 | 0.000 | 0.010 |
| trop_s | saliency | value_shuffled | 0.000 | -0.000 | 0.000 | 0.953 | 0.010 |
