# Figures

The report's figures, each saved as PNG (used by [REPORT.md](../REPORT.md)) and PDF. `scripts/make_figures.py` redraws every figure whose inputs exist. Fig. 20 is drawn by `scripts/damping_physics.py --figure`. The [v2 preprint](../papers/v2_preprint/) uses paper-style versions of Figs. 1, 2–3, 5–6, 18 and 13/13b, drawn by `scripts/make_figures.py --paper` into `papers/v2_preprint/figures/`.

| Figure | What it shows | Report section |
|---|---|---|
| [fig1_growth_spectrum](fig1_growth_spectrum.png) | Which ring modes of the uniform swing grow, and how fast (Floquet spectrum) | 3. Physics |
| [fig2_best_width](fig2_best_width.png) | The best box-car width follows the Floquet band edge, at seven amplitudes | 3. Physics |
| [fig3_label_sensitivity](fig3_label_sensitivity.png) | How precisely the initial state must be known, and which parts of the initial noise decide the site | 3. Physics |
| [fig4_baselines](fig4_baselines.png) | Physics baselines and the rebuilt v1 on the test pool | 5. Baselines |
| [fig5_receptive_field](fig5_receptive_field.png) | A site must see most of the ring | 6. Architecture |
| [fig6_data_efficiency](fig6_data_efficiency.png) | Respecting the ring's symmetry saves data | 6. Architecture |
| [fig7_architectures](fig7_architectures.png) | Phase 2: the architectures, on the same data and training budget | 6. Architecture |
| [fig8_noise_vs_time](fig8_noise_vs_time.png) | Sensor precision against how long you wait | 7. Inputs |
| [fig9_inputs](fig9_inputs.png) | Which measurements carry the information, without noise | 7. Inputs |
| [fig10_losses](fig10_losses.png) | Phase 4: loss functions | 8. Loss |
| [fig11_optimizers](fig11_optimizers.png) | Phase 5: optimizers and schedules | 9. Optimizer |
| [fig12_sparse_sensors](fig12_sparse_sensors.png) | Tracking every 2nd or 4th pendulum costs little; every 8th aliases | 7. Inputs |
| [fig13_noise_collapse](fig13_noise_collapse.png) | Waiting divides the sensor noise by the growth of the unstable modes | 7. Inputs |
| [fig13b_noise_avg_label](fig13b_noise_avg_label.png) | The same trade-off on the swing-averaged label | 7. Inputs |
| [fig14_leave_one_out](fig14_leave_one_out.png) | Phase 6: one choice taken out of the final model at a time | 10. Final model |
| [fig15_final_data_efficiency](fig15_final_data_efficiency.png) | The final model's error against the number of training runs | 10. Final model |
| [fig16_robustness](fig16_robustness.png) | Trained on the nominal chain, tested on shifted ones | 11. Robustness |
| [fig17_warning_tradeoff](fig17_warning_tradeoff.png) | Warning time against accuracy: watch later with a worse sensor | 11. Robustness |
| [fig18_label_jitter](fig18_label_jitter.png) | The first pendulum over 10% jitters around where the energy settles | 10. Final model |
| [fig19_avg_label_ablations](fig19_avg_label_ablations.png) | The same choices win on the swing-averaged label | 10. Final model |
| [fig20_damping](fig20_damping.png) | Damping decides whether the energy localizes, and how well v2 still predicts the site | 11. Robustness |
| [fig21_regularization](fig21_regularization.png) | Mirror images delay overfitting with little data | 10. Final model |
| [fig22_threshold](fig22_threshold.png) | Accuracy against the localization threshold, and how pinned each label is | 10. Final model |
| [fig23_settings](fig23_settings.png) | The hump label's knobs, localization time against the growth rate, the delay from rescaled offsets, the reach of linear theory, and models trained at other amplitudes and ring sizes | 10. Final model |
