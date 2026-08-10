Meta Review of Submission2158 by Area Chair rV3G
Meta Reviewby Area Chair rV3G26 Jul 2026, 16:02 (modified: 04 Aug 2026, 17:33)Senior Area Chairs, Area Chairs, Authors, Reviewers, Program ChairsRevisions
Metareview:
This paper proposes Q-Planning, which pairs a frozen large BC/VLA policy with a small off-policy Q-function: at inference, MPPI warm-started from BC samples and perturbed with temporally-smoothed noise proposes candidate action chunks that the Q-function scores, and a self-improvement loop then fine-tunes only the Q-function on deployment rollouts — including failures — while the BC policy is never updated. On LIBERO and RoboTwin the method improves over the frozen BC policy on four of five benchmarks offline, and online self-improvement lifts LIBERO-10 from 93% to 99% while updating only the Q-function. The core idea is elegant and timely: the actor/critic asymmetry (BC must imitate successes, but an off-policy Q-function can absorb any rollout) is a clean motivation for freezing the expensive policy and routing all deployment signal into a small critic. The paper is very well written, the temporal-smoothing contribution is genuine and well-supported (Table 1 shows vanilla MPPI slightly hurts while temporal smoothing helps, because jagged proposals leave the Q-function's support), and the ablations are clean and honest (warm-starting is essential: 93% vs 3.3%).

The reviewers diverge sharply (6/3/2), and my own reading lands at borderline for two reasons that the two critical reviewers raise and I find decisive. First, novelty is incremental: value-guided action selection over a frozen policy and off-policy Q-functions that learn from BC data plus online rollouts are an active, crowded area (e.g., value-guidance and BC-to-Q methods the paper itself cites), and Q-Planning largely composes established tools (Q-chunking, HL-Gauss, MPPI, BC warm-start); the genuinely new element is the temporal-smoothed proposal distribution. Second, and most important, the empirical case is missing the baselines needed to attribute the gains and support the headline claims. There is no Best-of-N comparison (BC samples ranked by the Q-function without MPPI), which is the natural test of whether MPPI adds anything over simple rejection sampling; no filtered-SFT comparison, which is the obvious way to test the "learns from failures" claim head-on; and no comparison to the closely-related value-guidance / imitation-bootstrapped-RL / BC-to-Q methods the paper cites. Compounding this, the offline gains are modest (e.g., +1.4pp on the LIBERO aggregate, with LIBERO-Object saturated), the evaluation is simulation-only, and the method adds substantial inference-time cost (on the order of 192 Q-evaluations per planning step) with no latency or real-time-feasibility analysis and no real-robot demonstration. A structural limitation also deserves attention: because exploration is confined to a temporally-smoothed Gaussian envelope around the BC mean, the system can only exploit high-reward action chunks once MPPI happens to sample them, and cannot reach behaviors far from the BC proposal.

Pre-Rebuttal Recommendation: Proceed to rebuttal: The paper has at least one score of weak accept or above, or the AC believes the paper warrants author response.
Key Issues for Rebuttal:
Missing baselines (most important). Add Best-of-N sampling (BC samples ranked by the Q-function, no MPPI) to isolate MPPI's contribution; add filtered SFT to test the "learns from failures" claim directly; and compare against at least one closely-related cited method (value guidance, imitation-bootstrapped RL, or BC-to-Q).
Latency / real-time feasibility. Report wall-clock cost per planning step given ~192 Q-evaluations, and discuss real-robot viability; a real-robot demonstration would substantially strengthen the paper.
Novelty positioning. Articulate precisely what is technically new relative to the value-guidance literature beyond training the Q-function online, and adjust claims accordingly.
Bootstrapping target. Justify the one-step-shifted bootstrap in Eq. 2 versus bootstrapping with MPPI-selected actions, and address the mismatch between the value being learned (BC's) and the policy actually deployed (the Q-guided planner).
Breadth of self-improvement. Extend the online self-improvement result beyond LIBERO-10 to the other suites / RoboTwin to show it is not specific to one benchmark.

Official Review of Submission2158 by Reviewer svm5
Official Reviewby Reviewer svm520 Jul 2026, 17:09 (modified: 04 Aug 2026, 17:34)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers, AuthorsRevisions
Summary:
This paper addresses the problem of improving the performance of a large robotic policy (e.g. VLA) without the computationally expensive steps of collecting more demonstration data or modifying the weights of the original model. The authors propose an online planning and RL fine-tuning approach: warm-start MPPI with actions from the pretrained model augmented with temporally smoothed gaussian noise to perform local exploration around the original policy's trajectory manifold. The new rollouts are added to the original dataset and used to train a lighter weight Q-function which, unlike the base model, can benefit from both success and failure data. At inference time, the method combines MPPI with the learned Q-function to score candidate action chunks and execute the one with highest value, resulting in test-time performance improvements over the base model.

Strengths:
The paper is well written and easy to follow. The authors lay out their contributions in simple terms and ablate each of their proposed components.
The proposed temporal smoothing component is novel and well-motivated. Naive noise sampling on the action chunks in order for MPPI to work results in lower performance than the base model because the sharp changes to the action chunks can quickly move the agent away from the optimal trajectory manifold. Temporally smoothing out the noise to stay near the manifold intuitively makes sense, enables local exploration around the trajectory manifold providing meaningful data for the Q-network to learn from, and results in decent performance gains over the base model.
Weaknesses:
The most significant weakness of this paper is the lack of comparison to baseline methods. The authors themselves cite a number of relevant methods ([1], [2]) which they don't mention or compare against in their experiments section. There are other works in a similar vein which were not cited ([3], [4]) that learn Q-functions from fine-tuning data collected from online rollouts, as the authors propose. The learned Q-functions are used to improve performance over the base policy without modifying the weights of the base policy. It is not clear to me why any of these methods were not used for comparison to the proposed method.

Regardless of making baseline comparisons or not, the authors' proposed method falls into an already crowded space of proposed efficient finetuning methods of large, pretrained base policies as referenced by [1]-[4], and the novelty of the contributions seems quite low. In section 4, the authors provide three core questions as the pillars of their analysis. Q1 and Q3 are not particularly interesting questions. From prior works, it should be very obvious by now that unless there were some catastrophic issues when training the Q-function for finetuning, that using a Q-function can improve the BC policy's offline performance and can enable self-improvement that the BC policy would not be able to do on its own. The only interesting and novel question is Q2, since to the best of my knowledge, the temporal smoothing aspect has not been thoroughly investigated in prior work. However, this alone is not enough to warrant a separate publication in my opinion.

[1] Nakamoto, Mitsuhiko, et al. "Steering your generalists: Improving robotic foundation models via value guidance." arXiv preprint arXiv:2410.13816 (2024).

[2] Wagenmaker, Andrew, et al. "Steering your diffusion policy with latent space reinforcement learning." arXiv preprint arXiv:2506.15799 (2025).

[3] Hu, Hengyuan, Suvir Mirchandani, and Dorsa Sadigh. "Imitation bootstrapped reinforcement learning." arXiv preprint arXiv:2311.02198 (2023).

[4] Dodeja, Lakshita, et al. "When Life Gives You BC, Make Q-functions: Extracting Q-values from Behavior Cloning for On-Robot Reinforcement Learning." arXiv preprint arXiv:2605.05172 (2026).

Questions For Authors:
Recent work has explored pre-trained values as guidance for robotic foundation models [29, 30, 31, 32, 33], but does not address how to use the value function to actively self-improve via online iteration.

This sentence seems to be the crux of the argument of why the authors' proposed method is technically novel compared to prior methods. However, to me this seems to be a difference in implementation detail. If one has access to an offline dataset, then this will be used to train the Q-function. If instead one has access to a simulator, then online rollouts will be used to train the Q-function. Can the authors explain why specifically RL finetuning in the online setting is worth investigating in a standalone paper?

Can the authors also explain how their method compares to [3] and [4] above, which do investigate the problem of RL finetuning in the online setting?

Limitations And Broader Impact:
n/a

Overall Score: 2: Reject. Clear reject. Fundamental issues in methodology, evaluation, or relevance that a rebuttal is unlikely to resolve.
Confidence Score: 4: High confidence. I am knowledgeable in this area and confident in my assessment.
Ethical Concerns: None
LLM Disclosure: No

Official Review of Submission2158 by Reviewer ZGT5
Official Reviewby Reviewer ZGT507 Jul 2026, 03:08 (modified: 04 Aug 2026, 17:34)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers, AuthorsRevisions
Summary:
The paper proposes Q-Planning, a method that uses off policy Q learning together with online MPPI planning in order to select actions that improve over a base foundation-model-based robotic policy. Q-Planning is evaluated on two well-established benchmarks: LIBERO and RoboTwin, showing improvement over the base policy.

Strengths:
Several design choices that are critical for the success of the method: initializing MPPI with the BC policy, using temporally-correlated noise and decoupling Q function and policy.
Arguing for benefits of off-policy RL in robotics with large foundation models is an important direction that is generally still open.
Weaknesses:
Limited evaluation: several ways in which the empirical evaluation could be improved:
real-world experiments would demonstrate that even when planning, latency is not an issue.
standard baselines: the paper does not compare Q-Planning with standard baselines like just simple Best of N sampling. Since all of the experiments are simulated, I would expect to also have a comparison with a method that adapts the BC model, and somehow highlights the advantage (e.g. compute saved, while achieving the same performance).
Another extremely simple yet powerful baseline is filtered SFT. The paper argues that Q-Planning leverages failures. An experiment comparing the two approaches, demonstrating exactly this claim, would make the paper stronger.
using more modern benchmark suites. LIBERO is relatively old.
Novelty: I am not familiar with state-of-the-art/very latest MPC + VLA literature, however the paper uses fairly standard tools.
Improvement is relatively limited. This could however improve on more challenging benchmarks.
Questions For Authors:
In lines 383-386 you mention that you don't run full integration of the flow matching model. How does your method compares to a standard BC model with the default number of flow-matching euler integration steps? Can you compare the latency of both approaches?
Can it run in real time? Since the method evaluates the Q function, which is in itself a relatively large model, in each iteration of MPPI, could you run Q-Planning on real robotic systems? An experiment analyzing the latency would be very useful to gain confidence. Do you run only the transformer decoder + head to predict Q values inside the MPPI loop?
What are the limits of this method? Can it improve performance on tasks in which the base policy does not generate even a single action chunk with high performance?
Did you try running your method on more modern benchmarks like MolmoSpaces [1]?
Technically, Q-Planning would benefit more from learning a Q function by bootstrapping with actions selected via MPPI, rather than actions shifted by one step as shown in equation (2). This is also what the authors in [2] show. Can you elaborate on your design choice here?
[1] Kim, Yejin, Wilbert Pumacay, Omar Rayyan, Max Argus, Winson Han, Eli VanderBilt, Jordi Salvador et al. "Molmospaces: A large-scale open ecosystem for robot navigation and manipulation." arXiv preprint arXiv:2602.11337 (2026).

[2] Li, Qiyang, Zhiyuan Paul Zhou, and Sergey Levine. "Reinforcement learning with action chunking." Advances in Neural Information Processing Systems 38 (2026): 55518-55553.

Minor: Line 12: "updating the BC" seems like a typo, doesn't read clearly. Line 12: "six iterations" can be misleading, since in each iteration Q-Planning collects many rollouts.

Limitations And Broader Impact:
While using techniques from off-policy RL is an important direction to post-training large VLAs for robots (as they can significantly improve sample efficiency), the paper's novelty is limited.

The paper discusses the core technical limitations of Q-Planning.

Overall Score: 3: Weak reject. Below the acceptance threshold. The paper has identifiable merit but significant weaknesses — e.g., missing key comparisons, unconvincing results, or incremental contribution.
Confidence Score: 4: High confidence. I am knowledgeable in this area and confident in my assessment.
Ethical Concerns: None
LLM Disclosure: No


Official Review of Submission2158 by Reviewer AzZT
Official Reviewby Reviewer AzZT30 Jun 2026, 11:57 (modified: 04 Aug 2026, 17:34)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers, AuthorsRevisions
Summary:
This work looks at combining behavior cloning (with a large model for robot control trained from successful teleoperation) with reinforcement learning (to enable learning from failed attempts). The key insight is that the value model can be trained on different data from the behavior model, allowing for adaptation of performance without having to change the BC model weights at all, thus avoiding costly updates (the value model is much smaller) as well as issues of forgetting and drift.

Given a large, frozen, BC model, the same data is used to initially train a value model. During execution, the BC model provides a ‘warm-start’ estimate of an action chunk to perform, which is then passed through MPPI with temporal smoothing, in effect exploring around the BC ‘mean’ with appropriate noise. Candidate action chunks are then rated by the value function, weighted together, and the resulting mean executed. Resulting performance is captured, and the new data sample (action + performance) used to update the value function only.

Strengths:
This work is in a very exciting area - lots of labs and companies are exploring combining imitation and reinforcement learning for robot control. As such, the issues it addresses, of how to improve system performance beyond the demonstrator in a time and data efficient fashion, are very apropos.

The work is solidly grounded, and the experiments support the claims. I also appreciate the ablation studies showing how each of the three contributions (BC warm start, temporal smoothing, Q-learning updates) contribute to a whole, self-improving system.

Weaknesses:
It’s unclear to me how the system can learn to exploit high-reward, but low-probability action chunks. Specifically, since the BC policy is fixed, it will output the same warm-start in a given scenario each time. It is then up to the randomness in the MPPI-temporal/smoothing step to enable both exploration (by introducing random changes) and exploitation (by weighing them by the learned Q function). It’s possible that a high-noise sample (unlikely under the Gaussian kernel) is generated far from the BC mean, that results in high-reward. The Q function will learn this, but cannot use this information until the MPPI step generates a similarly low-probability event in some future timestep.

That is, it appears the system is constrained to a Gaussian envelope of behaviors around the BC-generated action chunk.

My second concern has to do with the isolation of the visual and language parameters between the BC and Q models. I understand that this decision was made to allow the Q function to evolve beyond what the BC model has learned, but I’d like to see more exploration of what this means. Can the Q and BC models be attending to different features in the visual or language space? Might this situation be an indication that the BC model is ill-equipped to operate in whatever scenario it finds itself in?

Questions For Authors:
In addition to those above:

Could this approach be applied with initial models beyond just BC?  For example, I can train a chunk-generating model using both successful and unsuccessful/less-successful demonstrations, using contrastive or weighted learning, etc.

I found the section on Temporal-smoothed MPPI to be a little light.  Please provide a brief overview of MPPI and why it is not (or is) just adding noise to the BC-generated action chunk.
Limitations And Broader Impact:
What’s missing from the limitations section is what I described in the first part of ‘weaknesses’ - What are the limits of exploration imposed by this system?

There are no ethical concerns beyond those inherent in learning robot behavior from demonstration.

Real robot experiments would be awesome.

Overall Score: 6: Accept. A solid contribution that clearly meets the CoRL standard. Well-executed with convincing results, though there may be room for improvement in scope or presentation.
Confidence Score: 5: Expert. I have published extensively in this area and am certain of my assessment.
Ethical Concerns: None
LLM Disclosure: No

