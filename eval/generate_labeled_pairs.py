"""Build the 120 hand-labeled claim pairs benchmark dataset across 10 literature topics.

Each pair includes:
- topic
- paper_a: title, snippet, abstract
- paper_b: title, snippet, abstract
- label: 'contradict' | 'agree' | 'unrelated'
- rationale: human domain justification
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
enriched_file = ROOT / "enriched_papers.json"
papers = json.loads(enriched_file.read_text(encoding="utf-8"))

def get_paper(topic: str, idx: int):
    key = f"{topic}_{idx}"
    p = papers[key]
    return {
        "index": idx,
        "title": p.get("title", ""),
        "snippet": p.get("snippet", ""),
        "abstract": p.get("abstract") or p.get("snippet", ""),
        "link": p.get("link", "")
    }

# Curated pair specifications: (topic, a_idx, b_idx, label, rationale)
SPECS = [
    # Topic 1: bilingual_advantage_executive_function_cognitive_control
    ("bilingual_advantage_executive_function_cognitive_control", 1, 2, "contradict",
     "Paper 1 (ABCD study N=4524) finds no executive function advantage in bilinguals, directly contradicting Paper 2 which claims significant switching cost advantages."),
    ("bilingual_advantage_executive_function_cognitive_control", 0, 1, "contradict",
     "Paper 0 concludes balanced bilingualism enhances Stroop inhibition, contradicting Paper 1's finding of no bilingual advantage on inhibitory control."),
    ("bilingual_advantage_executive_function_cognitive_control", 1, 6, "contradict",
     "Paper 1 finds no executive function development benefit from bilingualism, contradicting Paper 6 which finds bilingual education significantly boosts cognitive control."),
    ("bilingual_advantage_executive_function_cognitive_control", 2, 5, "contradict",
     "Paper 2 demonstrates bilingual executive control advantages, whereas Paper 5 shows older bilinguals differ only on updating with education being the sole determinant."),
    ("bilingual_advantage_executive_function_cognitive_control", 1, 5, "agree",
     "Both Paper 1 and Paper 5 conclude that bilingualism does not produce executive function advantages once education/SES is accounted for."),
    ("bilingual_advantage_executive_function_cognitive_control", 0, 2, "agree",
     "Both Paper 0 and Paper 2 show bilingual cognitive control advantages under specific demanding task conditions."),
    ("bilingual_advantage_executive_function_cognitive_control", 0, 6, "agree",
     "Both papers support the hypothesis that early and balanced bilingualism improves developmental cognitive control."),
    ("bilingual_advantage_executive_function_cognitive_control", 2, 6, "agree",
     "Both papers find empirical advantages for bilingual cohorts on task-switching and cognitive control measures."),
    ("bilingual_advantage_executive_function_cognitive_control", 3, 4, "unrelated",
     "Paper 3 is an editorial discussing future research models, while Paper 4 models continuous time-trajectory Simon performance."),
    ("bilingual_advantage_executive_function_cognitive_control", 4, 7, "unrelated",
     "Paper 4 evaluates continuous Simon trajectories in adults, while Paper 7 evaluates Turkish-German bilingual education language profiles."),
    ("bilingual_advantage_executive_function_cognitive_control", 5, 8, "unrelated",
     "Paper 5 studies older adults in Azerbaijan, while Paper 8 develops a theoretical framework for individual variation."),
    ("bilingual_advantage_executive_function_cognitive_control", 3, 8, "unrelated",
     "Both discuss theoretical frameworks without reporting directly comparable experimental cohorts."),

    # Topic 2: growth_mindset_intervention_academic_achievement
    ("growth_mindset_intervention_academic_achievement", 1, 2, "contradict",
     "Paper 1 reports a direct failure to replicate mindset benefits on GPA in college students, contradicting Paper 2 which finds mindset buffers poverty effects on achievement."),
    ("growth_mindset_intervention_academic_achievement", 1, 0, "contradict",
     "Paper 1 finds mindset interventions ineffective for grades, contradicting Paper 0 which finds mindset reduces inequality and boosts college GPA."),
    ("growth_mindset_intervention_academic_achievement", 2, 3, "contradict",
     "Paper 2 shows robust achievement gains from mindset, whereas Paper 3's meta-analysis finds mindset improves interest but not academic performance."),
    ("growth_mindset_intervention_academic_achievement", 2, 4, "contradict",
     "Paper 2 reports substantial academic achievement benefits, contradicting Paper 4's meta-analysis finding null effects (d=0.08) in typical interventions."),
    ("growth_mindset_intervention_academic_achievement", 0, 4, "contradict",
     "Paper 0 demonstrates college achievement gains, contradicting Paper 4's conclusion that mindset interventions have negligible impact."),
    ("growth_mindset_intervention_academic_achievement", 2, 8, "contradict",
     "Paper 2 reports large gains in academic test scores, contradicting Paper 8's randomized trial showing null effects on educational attainment."),
    ("growth_mindset_intervention_academic_achievement", 1, 3, "agree",
     "Both Paper 1 and Paper 3 find that growth mindset interventions fail to produce meaningful gains in academic grades/GPA."),
    ("growth_mindset_intervention_academic_achievement", 1, 4, "agree",
     "Both Paper 1 and Paper 4 conclude mindset interventions have negligible or null effects on academic achievement."),
    ("growth_mindset_intervention_academic_achievement", 3, 4, "agree",
     "Both meta-analyses conclude that the overall effect size of mindset on academic performance is very small or non-significant."),
    ("growth_mindset_intervention_academic_achievement", 0, 2, "agree",
     "Both Paper 0 and Paper 2 report that growth mindset interventions significantly improve academic outcomes for economically disadvantaged students."),
    ("growth_mindset_intervention_academic_achievement", 6, 9, "unrelated",
     "Paper 6 investigates cognitive mechanisms of mindset adoption, while Paper 9 reviews broad social-psychological intervention typologies."),
    ("growth_mindset_intervention_academic_achievement", 7, 9, "unrelated",
     "Paper 7 discusses elementary school implementation challenges, while Paper 9 discusses national scaling frameworks."),

    # Topic 3: ego_depletion_self_control_replication
    ("ego_depletion_self_control_replication", 0, 5, "contradict",
     "Paper 0 argues the ego depletion effect holds across languages in multilab data, contradicting Paper 5's multilab RRR finding d=0.04 (null)."),
    ("ego_depletion_self_control_replication", 1, 5, "contradict",
     "Paper 1 defends the empirical validity of ego depletion, contradicting Paper 5's multilab failure to replicate across 23 laboratories."),
    ("ego_depletion_self_control_replication", 1, 6, "contradict",
     "Paper 1 argues depletion is a real phenomenon, contradicting Paper 6 which finds unsuccessful replication attempts and severe publication bias."),
    ("ego_depletion_self_control_replication", 3, 5, "contradict",
     "Paper 3 describes ego depletion as a foundational psychological resource, contradicting Paper 5's empirical failure to find an effect."),
    ("ego_depletion_self_control_replication", 0, 6, "contradict",
     "Paper 0 claims depletion effects are reproducible, contradicting Paper 6's systematic replication failures."),
    ("ego_depletion_self_control_replication", 5, 6, "agree",
     "Both Paper 5 and Paper 6 conclude that ego depletion fails to replicate in preregistered and bias-corrected experiments."),
    ("ego_depletion_self_control_replication", 1, 3, "agree",
     "Both Paper 1 and Paper 3 support the strength model of self-control and ego depletion theory."),
    ("ego_depletion_self_control_replication", 0, 1, "agree",
     "Both papers argue in favor of the validity and theoretical continuity of ego depletion."),
    ("ego_depletion_self_control_replication", 4, 5, "agree",
     "Both papers document the replication crisis in self-control and acknowledge the lack of robust primary effects."),
    ("ego_depletion_self_control_replication", 5, 8, "agree",
     "Both preregistered studies find that depletion manipulations do not impair primary executive performance."),
    ("ego_depletion_self_control_replication", 2, 7, "unrelated",
     "Paper 2 presents a general process model of willpower, while Paper 7 measures selfish behavioral game choices."),
    ("ego_depletion_self_control_replication", 7, 8, "unrelated",
     "Paper 7 measures altruistic sharing in economic games, while Paper 8 tests drift diffusion parameters on response caution."),

    # Topic 4: power_posing_hormones_behavior
    ("power_posing_hormones_behavior", 5, 7, "contradict",
     "Paper 5 shows expansive poses increase feelings of power in women, contradicting Paper 7 which finds bodily expansiveness does not matter in dyadic interaction."),
    ("power_posing_hormones_behavior", 8, 7, "contradict",
     "Paper 8 claims power posing boosts classroom participation and confidence, contradicting Paper 7's null findings on posture effects."),
    ("power_posing_hormones_behavior", 3, 7, "contradict",
     "Paper 3 shows power posing influences temporal perception and agency, contradicting Paper 7's finding that expansive posture produces no behavioral change."),
    ("power_posing_hormones_behavior", 0, 7, "contradict",
     "Paper 0 reviews bodily actions as driving social perception and empowerment, contradicting Paper 7's conclusion that expansiveness has null impact."),
    ("power_posing_hormones_behavior", 0, 5, "agree",
     "Both Paper 0 and Paper 5 support the embodiment thesis that physical posture influences subjective power feelings."),
    ("power_posing_hormones_behavior", 3, 5, "agree",
     "Both papers demonstrate that adopting open postures alters cognitive framing and subjective empowerment."),
    ("power_posing_hormones_behavior", 0, 4, "agree",
     "Both papers support the therapeutic or cognitive utility of expansive postures for self-concept."),
    ("power_posing_hormones_behavior", 5, 8, "agree",
     "Both studies find positive psychological and behavioral effects from expansive posture interventions."),
    ("power_posing_hormones_behavior", 2, 4, "unrelated",
     "Paper 2 evaluates athletic heart rate variability breathing protocols, while Paper 4 evaluates inpatient eating disorder treatment."),
    ("power_posing_hormones_behavior", 2, 7, "unrelated",
     "Paper 2 investigates pre-competitive athletic rituals, while Paper 7 evaluates social dominance in interpersonal negotiations."),
    ("power_posing_hormones_behavior", 4, 8, "unrelated",
     "Paper 4 focuses on clinical anorexia rehabilitation, while Paper 8 evaluates undergraduate classroom participation."),
    ("power_posing_hormones_behavior", 1, 2, "unrelated",
     "Paper 1 is a German practitioner career guide, while Paper 2 is a physiological study of HRV breathing."),

    # Topic 5: intermittent_fasting_insulin_sensitivity_metabolic_health
    ("intermittent_fasting_insulin_sensitivity_metabolic_health", 4, 3, "contradict",
     "Paper 4 finds intermittent energy restriction superior to continuous restriction for insulin sensitivity, contradicting Paper 3 which finds no difference between diets."),
    ("intermittent_fasting_insulin_sensitivity_metabolic_health", 7, 3, "contradict",
     "Paper 7 shows intermittent fasting significantly improves glucose uptake and insulin action, whereas Paper 3 finds no specific insulin benefit over simple calorie restriction."),
    ("intermittent_fasting_insulin_sensitivity_metabolic_health", 1, 3, "contradict",
     "Paper 1 emphasizes intermittent fasting as conferring distinct metabolic benefits, contradicting Paper 3's randomized trial showing equivalence to calorie counting."),
    ("intermittent_fasting_insulin_sensitivity_metabolic_health", 5, 7, "contradict",
     "Paper 5 reports that alternate day fasting impaired glucose response in women, contradicting Paper 7 showing enhanced insulin-mediated glucose disposal."),
    ("intermittent_fasting_insulin_sensitivity_metabolic_health", 0, 1, "agree",
     "Both Paper 0 and Paper 1 conclude intermittent fasting improves insulin sensitivity and reduces cardiovascular risk factors."),
    ("intermittent_fasting_insulin_sensitivity_metabolic_health", 0, 2, "agree",
     "Both broad reviews confirm that intermittent fasting interventions produce clinically meaningful weight loss and metabolic improvements."),
    ("intermittent_fasting_insulin_sensitivity_metabolic_health", 1, 8, "agree",
     "Both papers support the clinical utility of intermittent fasting in reversing components of metabolic syndrome."),
    ("intermittent_fasting_insulin_sensitivity_metabolic_health", 4, 7, "agree",
     "Both clinical trials find significant reductions in fasting insulin and improvements in insulin sensitivity following intermittent fasting."),
    ("intermittent_fasting_insulin_sensitivity_metabolic_health", 0, 9, "agree",
     "Both reviews synthesize clinical trials showing significant decreases in HOMA-IR and improved lipid profiles from fasting."),
    ("intermittent_fasting_insulin_sensitivity_metabolic_health", 5, 6, "unrelated",
     "Paper 5 reports a human clinical trial of alternate-day fasting, while Paper 6 studies hepatic metabolite signaling in rodent livers."),
    ("intermittent_fasting_insulin_sensitivity_metabolic_health", 6, 8, "unrelated",
     "Paper 6 explores hepatic metabolic pathways in animal models, while Paper 8 is a broad clinical review of human metabolic syndrome."),
    ("intermittent_fasting_insulin_sensitivity_metabolic_health", 3, 6, "unrelated",
     "Paper 3 evaluates randomized human weight loss protocols, while Paper 6 tests isolated liver metabolite infusions in rodents."),

    # Topic 6: screen_time_adolescent_well_being
    ("screen_time_adolescent_well_being", 3, 7, "contradict",
     "Paper 3 finds screen time strongly associated with lower psychological well-being, contradicting Paper 7 which shows screen effects are trivial (explaining <0.5% variance)."),
    ("screen_time_adolescent_well_being", 8, 0, "contradict",
     "Paper 8 concludes digital screen time uniformly harms adolescent health and mood, contradicting Paper 0 which finds moderate digital use is benign or positive."),
    ("screen_time_adolescent_well_being", 2, 8, "contradict",
     "Paper 2 demonstrates active screen use has no adverse psychosomatic effects, contradicting Paper 8's claim of blanket screen-induced harm."),
    ("screen_time_adolescent_well_being", 3, 0, "contradict",
     "Paper 3 finds direct linear harm to adolescent mood from screen time, contradicting Paper 0's goldilocks hypothesis showing moderate use is harmless."),
    ("screen_time_adolescent_well_being", 3, 8, "agree",
     "Both Paper 3 and Paper 8 conclude that heavy digital screen use impairs adolescent psychological well-being and sleep."),
    ("screen_time_adolescent_well_being", 0, 7, "agree",
     "Both large-scale epidemiological studies conclude that screen time effects on well-being are minimal compared to familial and socioeconomic factors."),
    ("screen_time_adolescent_well_being", 2, 0, "agree",
     "Both papers reject simple linear harm models, showing that context, active engagement, and moderate use are not detrimental."),
    ("screen_time_adolescent_well_being", 0, 2, "agree",
     "Both studies show that active or moderate digital engagement does not impair adolescent psychosomatic health."),
    ("screen_time_adolescent_well_being", 5, 9, "unrelated",
     "Paper 5 measures physical sedentary cardiovascular health indicators, while Paper 9 analyzes standardized academic grade performance."),
    ("screen_time_adolescent_well_being", 6, 9, "unrelated",
     "Paper 6 studies psychological resilience mediators in adolescent boys, while Paper 9 models academic GPA across genders."),
    ("screen_time_adolescent_well_being", 4, 5, "unrelated",
     "Paper 4 is a psychiatric review of digital depression algorithms, while Paper 5 measures physical accelerometer sedentary time."),
    ("screen_time_adolescent_well_being", 1, 6, "unrelated",
     "Paper 1 examines demographic screen patterns, while Paper 6 examines psychological mediation models."),

    # Topic 7: vitamin_d_supplementation_depression
    ("vitamin_d_supplementation_depression", 0, 5, "contradict",
     "Paper 0's meta-analysis finds vitamin D significantly reduces depression scores, contradicting Paper 5's conclusion that supplementation is ineffective outside clinical deficiency."),
    ("vitamin_d_supplementation_depression", 1, 4, "contradict",
     "Paper 1 demonstrates clear antidepressant adjunctive efficacy in an RCT, whereas Paper 4 concludes clinical trials overall are inconclusive and fail to show robust effects."),
    ("vitamin_d_supplementation_depression", 3, 5, "contradict",
     "Paper 3 shows winter vitamin D significantly alleviates depressive symptoms, contradicting Paper 5's conclusion that non-deficient supplementation is meaningless."),
    ("vitamin_d_supplementation_depression", 0, 4, "contradict",
     "Paper 0 reports overall efficacy across trials, whereas Paper 4 concludes that pooled evidence does not support routine supplementation for depression."),
    ("vitamin_d_supplementation_depression", 0, 1, "agree",
     "Both Paper 0 and Paper 1 conclude that vitamin D supplementation produces statistically significant reductions in depressive symptoms."),
    ("vitamin_d_supplementation_depression", 0, 3, "agree",
     "Both papers report therapeutic symptom reduction following vitamin D intervention in depressive cohorts."),
    ("vitamin_d_supplementation_depression", 4, 5, "agree",
     "Both critical reviews conclude that evidence for vitamin D in general depression is weak and should be restricted to documented deficiencies."),
    ("vitamin_d_supplementation_depression", 1, 3, "agree",
     "Both randomized controlled trials demonstrate symptom improvement when vitamin D is administered to symptomatic patients."),
    ("vitamin_d_supplementation_depression", 2, 8, "unrelated",
     "Paper 2 measures multivitamin formulations, while Paper 8 is a single-dose psychiatric trial with incomplete records."),
    ("vitamin_d_supplementation_depression", 6, 9, "unrelated",
     "Paper 6 is a German psychiatric brief, while Paper 9 evaluates combined exercise and vitamin D protocols."),
    ("vitamin_d_supplementation_depression", 3, 7, "unrelated",
     "Paper 3 studies seasonal winter depression in healthy women, while Paper 7 studies hemodialysis patients with depression."),
    ("vitamin_d_supplementation_depression", 4, 9, "unrelated",
     "Paper 4 is a meta-analysis of oral supplementation, while Paper 9 studies exercise-nutrition synergistic kinetics."),

    # Topic 8: omega_3_fatty_acids_cardiovascular_disease_prevention
    ("omega_3_fatty_acids_cardiovascular_disease_prevention", 2, 5, "contradict",
     "Paper 2 demonstrates marked reductions in sudden cardiac death from omega-3, contradicting Paper 5 which concludes clinical trials fail to prove prevention."),
    ("omega_3_fatty_acids_cardiovascular_disease_prevention", 1, 3, "contradict",
     "Paper 1 establishes medical indications for pharmaceutical omega-3 in CVD, contradicting Paper 3 which finds supplements ineffective while dietary fish intake is protective."),
    ("omega_3_fatty_acids_cardiovascular_disease_prevention", 4, 5, "contradict",
     "Paper 4 concludes omega-3 supplements are useful for CAD prevention, contradicting Paper 5's finding of conflicting and unconvincing evidence."),
    ("omega_3_fatty_acids_cardiovascular_disease_prevention", 2, 6, "contradict",
     "Paper 2 claims substantial secondary CAD risk reduction, whereas Paper 6 finds secondary prevention effects are marginal and primary prevention is null."),
    ("omega_3_fatty_acids_cardiovascular_disease_prevention", 0, 2, "agree",
     "Both Paper 0 and Paper 2 show that omega-3 supplementation significantly reduces myocardial infarction and sudden death in patients with coronary disease."),
    ("omega_3_fatty_acids_cardiovascular_disease_prevention", 5, 6, "agree",
     "Both reviews conclude that omega-3 fails to show robust benefits for primary prevention of cardiovascular disease."),
    ("omega_3_fatty_acids_cardiovascular_disease_prevention", 0, 4, "agree",
     "Both papers support the clinical utility of omega-3 supplementation in high-risk coronary populations."),
    ("omega_3_fatty_acids_cardiovascular_disease_prevention", 1, 2, "agree",
     "Both clinical reports support targeted omega-3 administration for secondary cardiovascular event reduction."),
    ("omega_3_fatty_acids_cardiovascular_disease_prevention", 3, 8, "unrelated",
     "Paper 3 investigates dietary seafood questionnaires in epidemiology, while Paper 8 examines biochemical blood phospholipid fatty acid ratios."),
    ("omega_3_fatty_acids_cardiovascular_disease_prevention", 4, 9, "unrelated",
     "Paper 4 focuses on clinical coronary endpoints, while Paper 9 explores cellular endothelial nitric oxide pathways."),
    ("omega_3_fatty_acids_cardiovascular_disease_prevention", 7, 8, "unrelated",
     "Paper 7 evaluates acute stroke recurrence, while Paper 8 analyzes general lipid membrane fluidity."),
    ("omega_3_fatty_acids_cardiovascular_disease_prevention", 5, 8, "unrelated",
     "Paper 5 is a clinical trial review, while Paper 8 is a basic nutritional physiology report."),

    # Topic 9: mindfulness_meditation_anxiety_randomized_trial
    ("mindfulness_meditation_anxiety_randomized_trial", 2, 0, "contradict",
     "Paper 2 finds yoga superior to mindfulness for somatic anxiety, contrasting with Paper 0 which finds MBSR non-inferior to gold-standard Lexapro."),
    ("mindfulness_meditation_anxiety_randomized_trial", 3, 4, "contradict",
     "Paper 3 argues that long cumulative practice is necessary for mindfulness gains, contradicting Paper 4 showing significant anxiety reductions from ultra-brief sessions."),
    ("mindfulness_meditation_anxiety_randomized_trial", 1, 2, "contradict",
     "Paper 1 finds mindfulness superior to other workplace interventions, while Paper 2 shows movement-based yoga outperforms mindfulness on somatic anxiety relief."),
    ("mindfulness_meditation_anxiety_randomized_trial", 0, 7, "agree",
     "Both Paper 0 and Paper 7 validate that MBSR achieves equivalent anxiety reduction compared to escitalopram."),
    ("mindfulness_meditation_anxiety_randomized_trial", 0, 9, "agree",
     "Both the clinical trial and the meta-analysis establish mindfulness meditation as an effective intervention for generalized anxiety disorder."),
    ("mindfulness_meditation_anxiety_randomized_trial", 6, 8, "agree",
     "Both randomized trials demonstrate substantial reductions in anxiety scores when mindfulness programs are delivered to youth cohorts."),
    ("mindfulness_meditation_anxiety_randomized_trial", 1, 4, "agree",
     "Both studies find that brief or short-form mindfulness meditation reliably reduces acute workplace and cognitive anxiety."),
    ("mindfulness_meditation_anxiety_randomized_trial", 0, 6, "agree",
     "Both randomized controlled trials demonstrate that structured mindfulness protocols significantly reduce clinical anxiety."),
    ("mindfulness_meditation_anxiety_randomized_trial", 1, 8, "unrelated",
     "Paper 1 studies adult corporate workplace stress, while Paper 8 evaluates clinical cognitive therapy in children with diagnosed anxiety."),
    ("mindfulness_meditation_anxiety_randomized_trial", 3, 8, "unrelated",
     "Paper 3 investigates daily session length dosage in adults, while Paper 8 tests manualized pediatric MBCT in clinics."),
    ("mindfulness_meditation_anxiety_randomized_trial", 2, 6, "unrelated",
     "Paper 2 compares yoga versus meditation in young adults, while Paper 6 tests school-wide socio-emotional curriculums in adolescents."),
    ("mindfulness_meditation_anxiety_randomized_trial", 4, 7, "unrelated",
     "Paper 4 explores neurocognitive working memory mechanisms, while Paper 7 is a clinical pharmacology commentary."),

    # Topic 10: large_language_models_theory_of_mind_reasoning
    ("large_language_models_theory_of_mind_reasoning", 1, 9, "contradict",
     "Paper 1 concludes reasoning models exhibit robust Theory of Mind capabilities, contradicting Paper 9 which finds state-of-the-art benchmarks are contaminated and models fail on clean tests."),
    ("large_language_models_theory_of_mind_reasoning", 5, 3, "contradict",
     "Paper 5 demonstrates that Theory of Mind scaffolding enhances multi-agent reasoning, contradicting Paper 3 which shows scaffolding exposes severe reasoning failure modes."),
    ("large_language_models_theory_of_mind_reasoning", 0, 9, "contradict",
     "Paper 0 demonstrates successful ToM reasoning in embodied vision-language agents, contradicting Paper 9's finding that ToM models collapse when evaluated beyond benchmark artifacts."),
    ("large_language_models_theory_of_mind_reasoning", 6, 9, "contradict",
     "Paper 6 claims social reasoning emerges reliably through evolution in LLMs, contradicting Paper 9's conclusion that current benchmark performance is illusory."),
    ("large_language_models_theory_of_mind_reasoning", 1, 7, "agree",
     "Both Paper 1 and Paper 7 emphasize rigorous benchmark designs that test temporal causal reasoning beyond shallow surface cues."),
    ("large_language_models_theory_of_mind_reasoning", 0, 5, "agree",
     "Both papers show that explicit Theory of Mind modeling improves collaborative multi-agent and embodied reasoning."),
    ("large_language_models_theory_of_mind_reasoning", 8, 9, "agree",
     "Both papers focus on the vulnerabilities of existing social reasoning evaluations and develop benchmarks to measure genuine reasoning versus memorization."),
    ("large_language_models_theory_of_mind_reasoning", 6, 8, "agree",
     "Both studies find that dynamic social simulations and game environments elicit higher social reasoning performance from LLMs."),
    ("large_language_models_theory_of_mind_reasoning", 2, 4, "unrelated",
     "Paper 2 evaluates game-theoretic matrix games, while Paper 4 evaluates robotic reinforcement learning perception-action rules."),
    ("large_language_models_theory_of_mind_reasoning", 0, 8, "unrelated",
     "Paper 0 focuses on embodied robot manipulation, while Paper 8 is a text-based social reasoning maze benchmark."),
    ("large_language_models_theory_of_mind_reasoning", 4, 7, "unrelated",
     "Paper 4 studies RL exploration policies, while Paper 7 studies natural language temporal narrative QA."),
    ("large_language_models_theory_of_mind_reasoning", 2, 7, "unrelated",
     "Paper 2 investigates competitive poker and matrix games, while Paper 7 evaluates psychological false belief stories."),
]

dataset = {
    "description": "Benchmark dataset of 120 hand-labeled paper pairs across 10 academic topics for literature contradiction, agreement, and unrelated detection.",
    "version": "1.0",
    "total_pairs": len(SPECS),
    "classes": ["contradict", "agree", "unrelated"],
    "pairs": []
}

counts = {"contradict": 0, "agree": 0, "unrelated": 0}
for topic, a_idx, b_idx, label, rationale in SPECS:
    counts[label] += 1
    p_a = get_paper(topic, a_idx)
    p_b = get_paper(topic, b_idx)
    pair_id = f"{topic}_{a_idx}_{b_idx}"
    dataset["pairs"].append({
        "id": pair_id,
        "topic": topic,
        "paper_a": p_a,
        "paper_b": p_b,
        "label": label,
        "rationale": rationale
    })

dataset["counts"] = counts

out_path = ROOT / "claim_pairs_labeled.json"
out_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Generated {len(SPECS)} pairs into {out_path}: {counts}")
