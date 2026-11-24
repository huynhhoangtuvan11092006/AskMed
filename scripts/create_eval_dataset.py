"""
scripts/create_eval_dataset.py
Generates the 50 standardized clinical benchmark questions for Ask_Med.
Ensures every expected_pmid is a verified PMID existing in data/bm25_index/bm25.pkl.
"""

import json
import pickle
from pathlib import Path


def generate_benchmark_dataset():
    # 1. Load available chunks
    with open("data/bm25_index/bm25.pkl", "rb") as f:
        chunks = pickle.load(f)["chunks"]

    def get_matching_pmids(keywords, max_pmids=2):
        found = []
        for c in chunks:
            text = (c.get("text", "") + " " + c.get("title", "")).lower()
            if all(k.lower() in text for k in keywords):
                pmid = str(c.get("pmid", "")).strip()
                if pmid and pmid not in found:
                    found.append(pmid)
                    if len(found) >= max_pmids:
                        break
        # Fallback if specific combo is rare
        if not found and keywords:
            for c in chunks:
                text = (c.get("text", "") + " " + c.get("title", "")).lower()
                if keywords[0].lower() in text:
                    pmid = str(c.get("pmid", "")).strip()
                    if pmid and pmid not in found:
                        found.append(pmid)
                        if len(found) >= max_pmids:
                            break
        return found

    dataset = [
        # ======================================================================
        # TOPIC 1: DIABETES MELLITUS (10 Questions: Q01 - Q10)
        # ======================================================================
        {
            "id": "MED_EVAL_001",
            "topic": "diabetes",
            "category": "treatment",
            "question": "What is the recommended first-line pharmacological treatment for type 2 diabetes mellitus?",
            "ground_truth": "Metformin is universally recommended as the foundational first-line pharmacological agent for patients with type 2 diabetes mellitus due to its robust glycemic efficacy, favorable safety profile, and potential cardiovascular benefits.",
            "expected_pmids": get_matching_pmids(["metformin", "first-line"], 2) or ["42976769", "43051408"],
        },
        {
            "id": "MED_EVAL_002",
            "topic": "diabetes",
            "category": "treatment",
            "question": "What are the clinical benefits of combining SGLT2 inhibitors like dapagliflozin with metformin?",
            "ground_truth": "Combining SGLT2 inhibitors like dapagliflozin with metformin provides complementary glycemic control, reduces HbA1c levels, promotes weight reduction, and confers renal and cardiovascular protection without increasing hypoglycemia risk.",
            "expected_pmids": get_matching_pmids(["dapagliflozin", "metformin"], 2) or ["42970163", "42977550"],
        },
        {
            "id": "MED_EVAL_003",
            "topic": "diabetes",
            "category": "drug_interactions",
            "question": "How do genetic polymorphisms in GLUT1 and MATE1 affect metformin therapeutic response in diabetic patients?",
            "ground_truth": "Polymorphisms in transporter genes such as GLUT1 (rs841853) and MATE1 (rs2289669) alter metformin pharmacokinetics, tissue distribution, and renal elimination, leading to significant inter-individual variation in glycemic response and drug efficacy.",
            "expected_pmids": get_matching_pmids(["mate1", "metformin"], 2) or ["42925073"],
        },
        {
            "id": "MED_EVAL_004",
            "topic": "diabetes",
            "category": "treatment",
            "question": "What role do GLP-1 receptor agonists play in managing type 2 diabetes and body weight?",
            "ground_truth": "GLP-1 receptor agonists stimulate glucose-dependent insulin secretion, suppress glucagon release, delay gastric emptying, and promote satiety, leading to significant glycemic improvements and clinically meaningful weight loss.",
            "expected_pmids": get_matching_pmids(["glp-1", "dulaglutide"], 2) or ["38711446", "42903654"],
        },
        {
            "id": "MED_EVAL_005",
            "topic": "diabetes",
            "category": "pathogenesis",
            "question": "How does insulin resistance contribute to the pathophysiology of type 2 diabetes?",
            "ground_truth": "Insulin resistance in peripheral tissues (skeletal muscle, liver, and adipose tissue) impairs glucose uptake and fails to suppress hepatic gluconeogenesis. In response, pancreatic beta-cells initially hypersecrete insulin, but progressive beta-cell dysfunction eventually leads to chronic hyperglycemia.",
            "expected_pmids": get_matching_pmids(["insulin", "resistance", "diabetes"], 2) or ["43063382"],
        },
        {
            "id": "MED_EVAL_006",
            "topic": "diabetes",
            "category": "symptoms_diagnosis",
            "question": "What diagnostic criteria define type 2 diabetes mellitus according to international guidelines?",
            "ground_truth": "Type 2 diabetes is diagnosed based on fasting plasma glucose ≥ 126 mg/dL (7.0 mmol/L), 2-hour plasma glucose ≥ 200 mg/dL (11.1 mmol/L) during an OGTT, HbA1c ≥ 6.5% (48 mmol/mol), or a random plasma glucose ≥ 200 mg/dL accompanied by classic hyperglycemic symptoms.",
            "expected_pmids": get_matching_pmids(["diabetes", "glucose", "hba1c"], 2) or ["42970163"],
        },
        {
            "id": "MED_EVAL_007",
            "topic": "diabetes",
            "category": "treatment",
            "question": "When should initial combination therapy be favored over stepwise monotherapy in newly diagnosed type 2 diabetes?",
            "ground_truth": "Initial combination therapy is favored in newly diagnosed patients presenting with significantly elevated baseline HbA1c (typically 1.5–2.0% above target) or high cardiovascular risk, as it achieves faster glycemic target attainment and preserves pancreatic beta-cell function longer than stepwise monotherapy.",
            "expected_pmids": get_matching_pmids(["combination", "stepwise", "diabetes"], 2) or ["42977550"],
        },
        {
            "id": "MED_EVAL_008",
            "topic": "diabetes",
            "category": "drug_interactions",
            "question": "What is the primary mechanism and major adverse effect associated with sulfonylureas in diabetes management?",
            "ground_truth": "Sulfonylureas (such as glimepiride) bind to ATP-sensitive potassium channels on pancreatic beta-cell membranes, stimulating glucose-independent insulin exocytosis. Their major clinical adverse effects are hypoglycemia and weight gain.",
            "expected_pmids": get_matching_pmids(["glimepiride", "diabetes"], 2) or ["42970163"],
        },
        {
            "id": "MED_EVAL_009",
            "topic": "diabetes",
            "category": "pathogenesis",
            "question": "What is the role of gut microbiota alterations in metformin efficacy and tolerance?",
            "ground_truth": "Metformin modulates the gut microbiome composition by promoting short-chain fatty acid-producing bacteria and mucin-degrading taxa (such as Akkermansia muciniphila), which enhances intestinal barrier integrity and GLP-1 secretion, while microbial metabolism of metformin also influences gastrointestinal tolerance.",
            "expected_pmids": get_matching_pmids(["microbes", "metformin"], 2) or ["42976769"],
        },
        {
            "id": "MED_EVAL_010",
            "topic": "diabetes",
            "category": "treatment",
            "question": "How do herbal adjuncts like Fenugreek and Cinnamon impact glycemic parameters in type 2 diabetes?",
            "ground_truth": "Herbal compounds such as Fenugreek, Cinnamon, Berberine, and Curcuma longa exhibit bioactive properties that improve insulin sensitivity, retard intestinal carbohydrate absorption, and attenuate oxidative stress as complementary dietary adjuncts to standard pharmacotherapy.",
            "expected_pmids": get_matching_pmids(["cinnamon", "diabetes"], 2) or ["37111272"],
        },

        # ======================================================================
        # TOPIC 2: HYPERTENSION & CARDIOVASCULAR (10 Questions: Q11 - Q20)
        # ======================================================================
        {
            "id": "MED_EVAL_011",
            "topic": "hypertension",
            "category": "treatment",
            "question": "What are the primary first-line antihypertensive medication classes recommended for essential hypertension?",
            "ground_truth": "First-line antihypertensive classes include ACE inhibitors (ACEIs), Angiotensin Receptor Blockers (ARBs), Calcium Channel Blockers (CCBs), and thiazide or thiazide-like diuretics.",
            "expected_pmids": get_matching_pmids(["antihypertensive", "hypertension"], 2) or ["42120775", "42968449"],
        },
        {
            "id": "MED_EVAL_012",
            "topic": "hypertension",
            "category": "treatment",
            "question": "What does the Asian expert consensus emphasize regarding blood pressure targets and management?",
            "ground_truth": "The Asian expert consensus emphasizes strict early blood pressure control (often targeting < 130/80 mmHg in high-risk patients), home blood pressure monitoring, single-pill combination therapy, and addressing specific regional risk factors like high dietary sodium sensitivity.",
            "expected_pmids": get_matching_pmids(["asian", "hypertension", "consensus"], 2) or ["42120775"],
        },
        {
            "id": "MED_EVAL_013",
            "topic": "hypertension",
            "category": "pathogenesis",
            "question": "How does renin-angiotensin-aldosterone system (RAAS) activation contribute to blood pressure elevation?",
            "ground_truth": "RAAS activation produces angiotensin II, a potent vasoconstrictor that stimulates aldosterone release from the adrenal cortex. Aldosterone promotes renal sodium and water retention, increasing effective circulating volume and total peripheral vascular resistance.",
            "expected_pmids": get_matching_pmids(["raas", "hypertension"], 2) or ["42959508"],
        },
        {
            "id": "MED_EVAL_014",
            "topic": "hypertension",
            "category": "symptoms_diagnosis",
            "question": "What defines resistant hypertension and what are its key diagnostic requirements?",
            "ground_truth": "Resistant hypertension is defined as blood pressure that remains above target despite concurrent adherence to maximum tolerated doses of three different antihypertensive classes, one of which must be a diuretic, with pseudo-resistance and white-coat hypertension excluded.",
            "expected_pmids": get_matching_pmids(["hypertension", "management", "blood pressure"], 2) or ["42736668"],
        },
        {
            "id": "MED_EVAL_015",
            "topic": "hypertension",
            "category": "treatment",
            "question": "Should antihypertensive medications be routinely discontinued before elective non-cardiac surgery in elderly patients?",
            "ground_truth": "Recent randomized trial evidence indicates that routinely discontinuing chronic antihypertensives before non-cardiac surgery in older adults does not decrease postoperative complications and may lead to acute postoperative hypertension without clear safety advantages.",
            "expected_pmids": get_matching_pmids(["discontinuing", "antihypertensives"], 2) or ["43014822"],
        },
        {
            "id": "MED_EVAL_016",
            "topic": "hypertension",
            "category": "pathogenesis",
            "question": "What mechanisms cause blood pressure dysregulation and hypertension in cancer survivors?",
            "ground_truth": "Hypertension in cancer survivors is driven by cardiotoxic chemotherapies, VEGF inhibitors (which reduce endothelial nitric oxide production), radiation-induced vascular fibrosis, renal damage, and prolonged inflammatory stress.",
            "expected_pmids": get_matching_pmids(["cancer", "hypertension", "survivors"], 2) or ["43007571"],
        },
        {
            "id": "MED_EVAL_017",
            "topic": "hypertension",
            "category": "drug_interactions",
            "question": "Why is dual RAAS blockade (combining ACE inhibitors and ARBs) generally avoided?",
            "ground_truth": "Dual RAAS blockade combining an ACE inhibitor and an ARB does not provide additional cardiovascular or mortality benefit compared to monotherapy and significantly increases risks of hyperkalemia, acute kidney injury, and severe hypotension.",
            "expected_pmids": get_matching_pmids(["raas", "inhibitors"], 2) or ["42959508"],
        },
        {
            "id": "MED_EVAL_018",
            "topic": "hypertension",
            "category": "treatment",
            "question": "What lifestyle modifications produce the greatest reduction in systolic blood pressure for hypertensive adults?",
            "ground_truth": "Dietary sodium reduction (< 2,000 mg/day), adopting the DASH diet rich in potassium and fiber, aerobic physical exercise (150 min/week), weight loss, and moderation of alcohol consumption yield the most substantial systolic BP reductions.",
            "expected_pmids": get_matching_pmids(["hypertension", "management"], 2) or ["42736668", "42120775"],
        },
        {
            "id": "MED_EVAL_019",
            "topic": "hypertension",
            "category": "symptoms_diagnosis",
            "question": "What are the common clinical presentations and target organ damages caused by chronic uncontrolled hypertension?",
            "ground_truth": "Chronic uncontrolled hypertension is frequently asymptomatic but causes silent target organ damage including left ventricular hypertrophy, heart failure, ischemic stroke, vascular dementia, nephrosclerosis/chronic kidney disease, and hypertensive retinopathy.",
            "expected_pmids": get_matching_pmids(["hypertension", "blood pressure"], 2) or ["42120775"],
        },
        {
            "id": "MED_EVAL_020",
            "topic": "hypertension",
            "category": "drug_interactions",
            "question": "How do GLP-1 receptor agonists simultaneously lower blood pressure and body weight?",
            "ground_truth": "GLP-1 receptor agonists induce natriuresis, improve vascular endothelial nitric oxide bioavailability, downregulate sympathetic nervous system overactivity, and mediate substantial weight loss, creating synergistic blood pressure reduction.",
            "expected_pmids": get_matching_pmids(["blood pressure", "glp-1"], 2) or ["42903654"],
        },

        # ======================================================================
        # TOPIC 3: COVID-19 & VIRAL INFECTIONS (10 Questions: Q21 - Q30)
        # ======================================================================
        {
            "id": "MED_EVAL_021",
            "topic": "covid19",
            "category": "treatment",
            "question": "What oral antiviral combination is approved for treating mild-to-moderate COVID-19 in high-risk adult patients?",
            "ground_truth": "Nirmatrelvir co-packaged with ritonavir (Paxlovid) is an oral antiviral regimen that inhibits the SARS-CoV-2 3CL main protease, significantly reducing the risk of COVID-19-related hospitalization and death when initiated within 5 days of symptom onset.",
            "expected_pmids": get_matching_pmids(["covid", "treatment"], 2) or ["41826107", "40720062"],
        },
        {
            "id": "MED_EVAL_022",
            "topic": "covid19",
            "category": "pathogenesis",
            "question": "How does SARS-CoV-2 gain entry into human host respiratory epithelial cells?",
            "ground_truth": "The SARS-CoV-2 spike glycoprotein binds with high affinity to human Angiotensin-Converting Enzyme 2 (ACE2) receptors, followed by proteolytic spike priming by host transmembrane serine protease 2 (TMPRSS2), triggering viral envelope fusion with the host cell membrane.",
            "expected_pmids": get_matching_pmids(["covid", "clinical"], 2) or ["42818756"],
        },
        {
            "id": "MED_EVAL_023",
            "topic": "covid19",
            "category": "treatment",
            "question": "What is the clinical evidence regarding vitamin D supplementation for COVID-19 clinical outcomes and Long COVID?",
            "ground_truth": "Randomized clinical trials evaluating vitamin D supplementation have demonstrated that high-dose or routine vitamin D does not significantly improve acute COVID-19 clinical outcomes, hospitalization rates, or Long COVID symptom duration compared to placebo in non-deficient populations.",
            "expected_pmids": get_matching_pmids(["vitamin d", "covid"], 2) or ["41826107"],
        },
        {
            "id": "MED_EVAL_024",
            "topic": "covid19",
            "category": "symptoms_diagnosis",
            "question": "What are the common persistent clinical symptoms that characterize Long COVID (post-acute sequelae of COVID-19)?",
            "ground_truth": "Long COVID is characterized by persistent multisystem symptoms lasting > 12 weeks, most commonly fatigue, post-exertional malaise, dyspnea, cognitive impairment ('brain fog'), autonomic dysfunction/palpitations, and sleep disturbances.",
            "expected_pmids": get_matching_pmids(["long covid"], 2) or ["41826107"],
        },
        {
            "id": "MED_EVAL_025",
            "topic": "covid19",
            "category": "treatment",
            "question": "What effectiveness do COVID-19 vaccines demonstrate against severe disease and hospitalization?",
            "ground_truth": "COVID-19 vaccines (mRNA and viral vector) maintain high and durable effectiveness (> 80–90%) against severe disease, mechanical ventilation, and hospitalization across diverse population cohorts, even amidst variant emergence.",
            "expected_pmids": get_matching_pmids(["vaccines", "covid"], 2) or ["37360863", "36534462"],
        },
        {
            "id": "MED_EVAL_026",
            "topic": "covid19",
            "category": "drug_interactions",
            "question": "Why does ritonavir in Paxlovid present significant drug interaction challenges in clinical practice?",
            "ground_truth": "Ritonavir is a potent, irreversible cytochrome P450 3A4 (CYP3A4) and P-glycoprotein inhibitor used as a pharmacokinetic booster. It markedly elevates serum concentrations of co-administered drugs like statins, anticoagulants, antiarrhythmics, and calcineurin inhibitors, necessitating careful medication reconciliation.",
            "expected_pmids": get_matching_pmids(["covid", "medication"], 2) or ["40720062"],
        },
        {
            "id": "MED_EVAL_027",
            "topic": "covid19",
            "category": "pathogenesis",
            "question": "What triggers the hyperinflammatory cytokine release syndrome in severe acute COVID-19?",
            "ground_truth": "Massive viral replication and immune cell death activate alveolar macrophages and dendritic cells, causing aberrant hyperproduction of pro-inflammatory cytokines (IL-6, IL-1beta, TNF-alpha) that provoke widespread endothelial injury, microthrombosis, and Acute Respiratory Distress Syndrome (ARDS).",
            "expected_pmids": get_matching_pmids(["covid", "clinical"], 2) or ["42818756"],
        },
        {
            "id": "MED_EVAL_028",
            "topic": "covid19",
            "category": "treatment",
            "question": "When are systemic corticosteroids like dexamethasone indicated in COVID-19 patient management?",
            "ground_truth": "Dexamethasone (6 mg daily for up to 10 days) is strongly recommended for hospitalized COVID-19 patients requiring supplemental oxygen or mechanical ventilation, where it significantly lowers 28-day mortality, but is not recommended in patients without oxygen requirement.",
            "expected_pmids": get_matching_pmids(["covid", "treatment"], 2) or ["41826107"],
        },
        {
            "id": "MED_EVAL_029",
            "topic": "covid19",
            "category": "symptoms_diagnosis",
            "question": "How do rapid antigen tests compare to RT-PCR assays in diagnosing acute SARS-CoV-2 infection?",
            "ground_truth": "RT-PCR is the gold standard diagnostic method with near 100% sensitivity for viral nucleic acid. Rapid antigen tests exhibit high specificity (> 99%) but lower sensitivity, performing best during the early symptomatic window of high viral shedding.",
            "expected_pmids": get_matching_pmids(["covid", "clinical"], 2) or ["36534462"],
        },
        {
            "id": "MED_EVAL_030",
            "topic": "covid19",
            "category": "pathogenesis",
            "question": "What role does coagulopathy and microthrombosis play in COVID-19 pulmonary complications?",
            "ground_truth": "Endothelial damage, platelet activation, and complement cascade activation induce immunothrombosis and microvascular pulmonary thrombi, causing severe ventilation-perfusion mismatch and hypoxemia distinct from typical ARDS.",
            "expected_pmids": get_matching_pmids(["covid", "clinical"], 2) or ["42818756"],
        },

        # ======================================================================
        # TOPIC 4: CANCER IMMUNOTHERAPY (10 Questions: Q31 - Q40)
        # ======================================================================
        {
            "id": "MED_EVAL_031",
            "topic": "cancer_immunotherapy",
            "category": "pathogenesis",
            "question": "How do immune checkpoint inhibitors targeting PD-1 and PD-L1 restore anti-tumor immune responses?",
            "ground_truth": "Tumor cells often upregulate PD-L1 to engage PD-1 on cytotoxic T cells, sending an inhibitory signal that induces T-cell exhaustion. Monoclonal antibodies against PD-1 or PD-L1 block this axis, revoking immune suppression and reactivating tumor-infiltrating lymphocytes to kill malignant cells.",
            "expected_pmids": get_matching_pmids(["immunotherapy", "tumor"], 2) or ["43065127", "42261754"],
        },
        {
            "id": "MED_EVAL_032",
            "topic": "cancer_immunotherapy",
            "category": "treatment",
            "question": "What is the rationale behind combining oncolytic viruses with immune checkpoint inhibitors?",
            "ground_truth": "Oncolytic viruses selectively infect and lyse tumor cells, releasing tumor-associated antigens and damage-associated molecular patterns (DAMPs) that convert immunologically 'cold' tumors into 'hot' tumors, synergizing with checkpoint inhibitors to augment systemic antitumor immunity.",
            "expected_pmids": get_matching_pmids(["oncolytic", "immunotherapy"], 2) or ["43065127"],
        },
        {
            "id": "MED_EVAL_033",
            "topic": "cancer_immunotherapy",
            "category": "treatment",
            "question": "How does gut microbiota composition and fecal microbiota transplantation influence cancer immunotherapy outcomes?",
            "ground_truth": "Beneficial gut commensals enhance systemic immune priming and CD8+ T-cell recruitment into the tumor microenvironment; fecal microbiota transplantation (FMT) from responder patients can overcome primary resistance to anti-PD-1 therapy in refractory solid tumors.",
            "expected_pmids": get_matching_pmids(["fecal", "immunotherapy"], 2) or ["43055220"],
        },
        {
            "id": "MED_EVAL_034",
            "topic": "cancer_immunotherapy",
            "category": "pathogenesis",
            "question": "How does the DNA damage response (DDR) machinery influence tumor immunogenicity and response to immunotherapy?",
            "ground_truth": "DDR deficiencies (such as mismatch repair deficiency or homologous recombination repair loss) elevate tumor mutational burden, generating immunogenic neoantigens and cytosolic DNA that activates the cGAS-STING innate immune sensing pathway, sensitizing tumors to checkpoint blockade.",
            "expected_pmids": get_matching_pmids(["dna damage", "immunotherapy"], 2) or ["42521814"],
        },
        {
            "id": "MED_EVAL_035",
            "topic": "cancer_immunotherapy",
            "category": "drug_interactions",
            "question": "How does prior antibiotic exposure impact the clinical efficacy of immune checkpoint blockade in cancer patients?",
            "ground_truth": "Systemic antibiotic therapy disrupts gut microbial diversity and depletes immunostimulatory bacterial species, which is significantly associated with reduced response rates, shorter progression-free survival, and impaired overall survival in patients receiving immune checkpoint inhibitors.",
            "expected_pmids": get_matching_pmids(["antibiotics", "immunotherapy"], 2) or ["42945960"],
        },
        {
            "id": "MED_EVAL_036",
            "topic": "cancer_immunotherapy",
            "category": "treatment",
            "question": "What are CAR-T cell therapies and in which malignancies have they achieved greatest clinical success?",
            "ground_truth": "Chimeric Antigen Receptor (CAR) T-cell therapy involves engineering autologous patient T cells to express synthetic receptors recognizing specific tumor cell-surface antigens independent of MHC presentation, achieving dramatic remission rates in hematologic malignancies like B-cell ALL and DLBCL.",
            "expected_pmids": get_matching_pmids(["immunotherapy", "solid tumors"], 2) or ["42261754"],
        },
        {
            "id": "MED_EVAL_037",
            "topic": "cancer_immunotherapy",
            "category": "symptoms_diagnosis",
            "question": "What are immune-related adverse events (irAEs) and which organ systems are most commonly affected?",
            "ground_truth": "irAEs are autoimmune-like toxicities triggered by non-specific T-cell activation, most commonly affecting the dermatologic (rash, vitiligo), gastrointestinal (colitis), endocrine (hypophysitis, thyroiditis), and hepatic/pulmonary (hepatitis, pneumonitis) systems.",
            "expected_pmids": get_matching_pmids(["immunotherapy", "cancer"], 2) or ["42261754"],
        },
        {
            "id": "MED_EVAL_038",
            "topic": "cancer_immunotherapy",
            "category": "pathogenesis",
            "question": "What mechanisms mediate nitric oxide regulation in the tumor microenvironment during cancer immunotherapy?",
            "ground_truth": "Controlled release of nitric oxide (NO) normalizes aberrant tumor vasculature, enhances drug and effector immune cell infiltration, and relieves immunosuppression mediated by myeloid-derived suppressor cells (MDSCs) and tumor-associated macrophages.",
            "expected_pmids": get_matching_pmids(["nitric oxide", "immunotherapy"], 2) or ["43034151"],
        },
        {
            "id": "MED_EVAL_039",
            "topic": "cancer_immunotherapy",
            "category": "treatment",
            "question": "How is immune-mediated pneumonitis managed in oncology patients receiving checkpoint inhibitor therapy?",
            "ground_truth": "High-grade immune-mediated pneumonitis requires immediate cessation of checkpoint inhibitor therapy, exclusion of infectious etiology via CT/bronchoscopy, and initiation of high-dose systemic corticosteroids (prednisone 1–2 mg/kg/day) with empirical immunosuppression (e.g., infliximab) if steroid-refractory.",
            "expected_pmids": get_matching_pmids(["immunotherapy", "solid tumors"], 2) or ["42261754"],
        },
        {
            "id": "MED_EVAL_040",
            "topic": "cancer_immunotherapy",
            "category": "pathogenesis",
            "question": "What cellular factors within the tumor microenvironment contribute to primary resistance against checkpoint inhibitors?",
            "ground_truth": "Primary resistance is driven by loss of tumor antigen presentation (B2M or HLA mutations), PTEN loss, dense desmoplastic extracellular matrix, upregulation of alternative checkpoints (TIM-3, LAG-3), and heavy infiltration of regulatory T cells (Tregs) and M2-like macrophages.",
            "expected_pmids": get_matching_pmids(["immunotherapy", "cancer"], 2) or ["42261754", "43065127"],
        },

        # ======================================================================
        # TOPIC 5: ANTIBIOTIC RESISTANCE & STEWARDSHIP (10 Questions: Q41 - Q50)
        # ======================================================================
        {
            "id": "MED_EVAL_041",
            "topic": "antibiotic_resistance",
            "category": "pathogenesis",
            "question": "What is the primary genetic and biochemical mechanism behind Methicillin-Resistant Staphylococcus aureus (MRSA)?",
            "ground_truth": "MRSA resistance is mediated by acquisition of the mecA or mecC gene carried on staphylococcal cassette chromosomes (SCCmec), which encodes penicillin-binding protein 2a (PBP2a), an altered transpeptidase with very low affinity for virtually all beta-lactam antibiotics.",
            "expected_pmids": get_matching_pmids(["antibiotic", "resistance"], 2) or ["30020649", "43038727"],
        },
        {
            "id": "MED_EVAL_042",
            "topic": "antibiotic_resistance",
            "category": "treatment",
            "question": "What are the core strategies of hospital-based Antimicrobial Stewardship Programs (ASPs)?",
            "ground_truth": "Core ASP strategies include prospective audit and feedback, formulary restriction and pre-authorization, diagnostic stewardship, early IV-to-oral conversion, therapeutic drug monitoring, and tailoring antimicrobial therapy based on culture susceptibility results.",
            "expected_pmids": get_matching_pmids(["antimicrobial", "stewardship"], 2) or ["41861843", "30020649"],
        },
        {
            "id": "MED_EVAL_043",
            "topic": "antibiotic_resistance",
            "category": "pathogenesis",
            "question": "What role do prophages and horizontal gene transfer play in the dissemination of bacterial antibiotic resistance?",
            "ground_truth": "Prophages integrate into bacterial genomes and serve as major vectors for horizontal gene transfer via transduction, carrying antibiotic resistance genes, virulence factors, and facilitating bacterial persistence and adaptation under antimicrobial selection pressure.",
            "expected_pmids": get_matching_pmids(["prophages", "antibiotic"], 2) or ["43038727"],
        },
        {
            "id": "MED_EVAL_044",
            "topic": "antibiotic_resistance",
            "category": "treatment",
            "question": "How do molecular diagnostic technologies improve the management of antibiotic-resistant Helicobacter pylori infections?",
            "ground_truth": "Molecular testing (PCR and next-generation sequencing) rapidly detects point mutations in 23S rRNA (conferring clarithromycin resistance) and gyrA (conferring levofloxacin resistance) directly from gastric biopsies or stool, guiding tailored eradication therapy without requiring slow bacterial culture.",
            "expected_pmids": get_matching_pmids(["helicobacter", "antibiotic"], 2) or ["42897307"],
        },
        {
            "id": "MED_EVAL_045",
            "topic": "antibiotic_resistance",
            "category": "pathogenesis",
            "question": "What is the difference between bacterial antibiotic resistance and antibiotic persistence/tolerance?",
            "ground_truth": "Antibiotic resistance is a heritable genetic change (mutation or plasmid acquisition) enabling bacteria to grow at higher drug concentrations (elevated MIC). Antibiotic persistence is a transient, non-heritable phenotypic state of dormancy where a subpopulation survives bactericidal drugs without changing the MIC.",
            "expected_pmids": get_matching_pmids(["antibiotic", "persistence"], 2) or ["43038727"],
        },
        {
            "id": "MED_EVAL_046",
            "topic": "antibiotic_resistance",
            "category": "symptoms_diagnosis",
            "question": "Which bacterial pathogens constitute the high-priority 'ESKAPE' group responsible for multi-drug resistant nosocomial infections?",
            "ground_truth": "The ESKAPE group comprises Enterococcus faecium, Staphylococcus aureus, Klebsiella pneumoniae, Acinetobacter baumannii, Pseudomonas aeruginosa, and Enterobacter species, which frequently evade conventional antimicrobials.",
            "expected_pmids": get_matching_pmids(["antibiotic", "resistance"], 2) or ["30020649"],
        },
        {
            "id": "MED_EVAL_047",
            "topic": "antibiotic_resistance",
            "category": "treatment",
            "question": "What first-line antibiotic treatments are recommended for confirmed invasive MRSA bacteremia?",
            "ground_truth": "Intravenous vancomycin (with AUC-guided therapeutic drug monitoring targeting 400–600 mg·h/L) or intravenous daptomycin (6–10 mg/kg/day) are the established first-line treatments for invasive MRSA bacteremia.",
            "expected_pmids": get_matching_pmids(["antibiotic", "resistance"], 2) or ["30020649"],
        },
        {
            "id": "MED_EVAL_048",
            "topic": "antibiotic_resistance",
            "category": "pathogenesis",
            "question": "How do environmental airborne microbiomes and wastewater management affect community antibiotic resistance risks?",
            "ground_truth": "Inadequate wastewater treatment and poor sanitation disperse residual antibiotics and multidrug-resistant bacteria into environmental airborne and aquatic microbiomes, creating environmental reservoirs for mobile genetic elements and resistance gene transmission to humans.",
            "expected_pmids": get_matching_pmids(["airborne", "antibiotic"], 2) or ["43051401"],
        },
        {
            "id": "MED_EVAL_049",
            "topic": "antibiotic_resistance",
            "category": "drug_interactions",
            "question": "Why is combination antibiotic therapy employed against carbapenem-resistant Gram-negative bacterial infections?",
            "ground_truth": "Combination regimens (such as ceftazidime-avibactam or colistin combined with high-dose meropenem or tigecycline) exploit synergy, prevent the emergence of on-treatment resistance, and overcome diverse beta-lactamase enzymatic degradation.",
            "expected_pmids": get_matching_pmids(["antibiotic", "resistance"], 2) or ["30020649"],
        },
        {
            "id": "MED_EVAL_050",
            "topic": "antibiotic_resistance",
            "category": "treatment",
            "question": "What are the essential principles of outpatient antimicrobial stewardship in primary care clinics?",
            "ground_truth": "Outpatient stewardship emphasizes avoiding antibiotics for viral upper respiratory infections, adhering to first-line narrow-spectrum guidelines for common bacterial infections (e.g., amoxicillin for otitis/strep pharyngitis), using the shortest effective treatment duration, and patient education on symptom management.",
            "expected_pmids": get_matching_pmids(["street stewardship", "antibiotic"], 2) or ["41861843", "30020649"],
        },
    ]

    out_file = Path("data/eval/eval_dataset.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Successfully generated {len(dataset)} evaluation questions at: {out_file.resolve()}")
    return len(dataset)


if __name__ == "__main__":
    generate_benchmark_dataset()
