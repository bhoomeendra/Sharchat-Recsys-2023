Sharchat Recsys 2023 Challange - Recommendations Squad

List of the expreiment results:

|        xgb_all_feat        | 5.968829287 | 6.37396  |          24/05/2023 05:09          |                    Hyper parameter turing with optuna                    |
|:--------------------------:|-------------|----------|:----------------------------------:|:------------------------------------------------------------------------:|
|      f1_xgb_catb_best      |             | 6.375778 |          27/05/2023 15:33          |                 Combined XGB and Catboost with f1 formula                |
|   xgb_all_feat_xgb_chain   | 5.99        | 6.385457 |          25/05/2023 16:06          |                     Xgb on top of another Xgb model                      |
|      catboost_all_feat     | 5.739692312 | 6.461268 |          25/05/2023 11:42          |              Catboost model with all the feature with optuna             |
| xgb_stacked_kfold_logistic | Nan         | 6.611657 |          05/06/2023 16:34          |                   XGB Stacked kflod Logistic Regression                  |
|   xgb_calibrated_logistic  |             | 6.61368  |          26/05/2023 14:55          |                  Calibrated XGB with logistic regression                 |
|  xgb_stack_all_cat_num_xgb | 6.026       | 6.64367  |          25/05/2023 06:20          |             Stack all the above row 1,2 and 3 optuna with xgb            |
|     xgb_num_cat_all_avg    |             | 6.690628 |          25/05/2023 16:22          |                  Simple average of all the xgb of row 1                  |
|        xgb_cat_feat        | 6.248026244 | 6.927283 |          24/05/2023 05:18          | Hyper parameter turing with optuna while using only Categorical Features |
|     xgb_float_all_feat     | 5.79        | 7.254    |          27/05/2023 21:11          |           Converted all the categorical values to probablites            |
|        xgb_num_feat        | 6.245909454 | 7.401632 | 24/05/2023 05:18 ,24/05/2023 05:19 |  Hyper parameter turing with optuna while using only Numerical Features  |
|    xgb_stack_all_cat_num   | 7.68580268  | 9.910395 |          25/05/2023 04:10          | Stack all the above row 1,2 and 3 optuna with logistice regression model |
|        xgb_full_data       | Nan         |          | 27/05/2023 22:29                   |    Validation data is also mixed with whole data but dates is not used   |
|                            |             |          |                                    |                                                                          |
|                            |             |          |                                    |                                                                          |