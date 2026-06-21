# Robot-Order FK Reset State/Action Distribution Diagnostic

## Scope

This diagnostic reads the existing same-seed full-evaluation traces for the robot-order FK-repaired local PPO checkpoint. It does not launch Isaac Sim, does not train, and does not claim paper-level tracking.

## Key Metrics

- `baseline_step0_body_error`: `43.294166564941406`
- `target_refresh_step0_body_error`: `0.26462867856025696`
- `target_refresh_step0_joint_vel`: `17.829124450683594`
- `baseline_step0_joint_vel`: `0.0`
- `target_refresh_first5_action_abs_mean_delta`: `0.07184725403785702`
- `target_refresh_post_step0_done_rate_delta`: `0.047659854760906034`
- `target_refresh_ee_body_pos_termination_fraction_delta`: `0.0478825904055184`

## Interpretation

- Primary bottleneck: No-advance target refresh removes the stale step-0 body target, but it exposes or creates a large initial joint-velocity/action transient and still worsens post-step0 done rate. The current teacher should not be used as the final DAgger/VAE/diffusion data source.
- Recommended next experiment: Patch reset-state and reset-action consistency before rerunning PPO: align initial joint velocities and last-action observations with the refreshed target, then rerun the full robot-order FK task eval and only proceed to full PPO if post-step0 done rate improves.

## Window Summary

| variant | window | done_rate | reward_mean | action_abs_mean | body_pos | joint_vel | body_ang_vel |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | step0 | 1.0 | 0.020213397219777107 | 0.1424887776374817 | 43.294166564941406 | 0.0 | 1.5725736618041992 |
| baseline | first5 | 0.30166015625 | 0.029593997076153754 | 0.2688242018222809 | 8.815188476443291 | 13.874491691589355 | 3.03814492225647 |
| baseline | post_step0 | 0.17552236262583892 | 0.020735592350063708 | 0.5594908988115771 | 0.21707132513691116 | 16.14195649895892 | 3.4088612610861757 |
| baseline | all | 0.1782798129180602 | 0.02073384587805606 | 0.5580962428879179 | 0.36114187777839774 | 16.087970022373774 | 3.402719830988243 |
| reset_command_warmup_seed_matched | step0 | 0.291015625 | 0.03187435865402222 | 0.14281846582889557 | 0.26491862535476685 | 17.904268264770508 | 3.2565364837646484 |
| reset_command_warmup_seed_matched | first5 | 0.154296875 | 0.03058817274868488 | 0.3405051052570343 | 0.21284139454364776 | 18.185510635375977 | 3.6958789348602297 |
| reset_command_warmup_seed_matched | post_step0 | 0.2213044646602349 | 0.020245333289690064 | 0.5190905982115924 | 0.24385091007355875 | 15.785051093005494 | 3.3823959963433694 |
| reset_command_warmup_seed_matched | all | 0.22153761235367894 | 0.020284226351109238 | 0.5178321629862322 | 0.24392137065978353 | 15.792138775854207 | 3.3819750615186916 |
| reset_target_refresh_no_advance | step0 | 0.29052734375 | 0.03188012167811394 | 0.14278434216976166 | 0.26462867856025696 | 17.829124450683594 | 3.2580718994140625 |
| reset_target_refresh_no_advance | first5 | 0.1537109375 | 0.030692277103662492 | 0.3406714558601379 | 0.21265828013420104 | 18.083859634399413 | 3.6945339679718017 |
| reset_target_refresh_no_advance | post_step0 | 0.22318221738674496 | 0.020134724770366347 | 0.5147821041661621 | 0.24175233633926252 | 15.789847998011032 | 3.3758608502829635 |
| reset_target_refresh_no_advance | all | 0.22340745192307693 | 0.020174007034271857 | 0.5135379644939333 | 0.24182884584501835 | 15.796668320595222 | 3.3754669073034687 |

## Deltas Versus Baseline

| comparison | window | done_rate_delta | action_abs_mean_delta | body_pos_delta | joint_vel_delta |
|---|---:|---:|---:|---:|---:|
| reset_command_warmup_seed_matched_minus_baseline | step0 | -0.708984375 | 0.0003296881914138794 | -43.02924793958664 | 17.904268264770508 |
| reset_command_warmup_seed_matched_minus_baseline | first5 | -0.14736328124999998 | 0.07168090343475342 | -8.602347081899643 | 4.311018943786621 |
| reset_command_warmup_seed_matched_minus_baseline | post_step0 | 0.04578210203439598 | -0.04040030059998467 | 0.026779584936647588 | -0.3569054059534249 |
| reset_command_warmup_seed_matched_minus_baseline | all | 0.04325779943561875 | -0.04026407990168568 | -0.1172205071186142 | -0.29583124651956716 |
| reset_target_refresh_no_advance_minus_baseline | step0 | -0.70947265625 | 0.00029556453227996826 | -43.02953788638115 | 17.829124450683594 |
| reset_target_refresh_no_advance_minus_baseline | first5 | -0.14794921874999997 | 0.07184725403785702 | -8.60253019630909 | 4.209367942810058 |
| reset_target_refresh_no_advance_minus_baseline | post_step0 | 0.047659854760906034 | -0.044708794645414995 | 0.02468101120235136 | -0.35210850094788704 |
| reset_target_refresh_no_advance_minus_baseline | all | 0.045127639005016734 | -0.04455827839398463 | -0.11931303193337939 | -0.2913017017785524 |
