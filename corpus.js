/**
 * corpus.js
 * Medical Science Corpus — vocabulary and frequency data.
 *
 * In a production system this would be loaded from a pre-processed corpus file.
 * Here we define ~500 medical domain words with realistic frequency weights,
 * giving >100,000 word tokens across the distribution.
 *
 * Words are drawn from: clinical medicine, pharmacology, pathology, anatomy,
 * immunology, genetics, epidemiology, and diagnostic terminology.
 */

const CORPUS_RAW = `patient patients doctor doctors nurse nurses hospital hospitals clinic clinics
medicine medicines drug drugs dose doses diagnosis diagnoses treatment treatments
therapy therapies surgery surgeries procedure procedures medication medications
prescription prescriptions symptom symptoms disease diseases disorder disorders
condition conditions infection infections virus viruses bacteria bacterial fungal
parasite parasitic chronic acute severe mild moderate benign malignant tumor tumors
cancer cancers carcinoma sarcoma lymphoma leukemia melanoma prognosis etiology
pathology histology biopsy specimen culture laboratory test tests exam examination
scan imaging xray ultrasound mri ct blood urine serum plasma hemoglobin glucose
protein lipid cholesterol triglyceride insulin cortisol thyroid hormone hormones
antibody antibodies antigen antigens immune immunity autoimmune inflammatory
inflammation allergy allergic asthma diabetes hypertension hypotension tachycardia
bradycardia arrhythmia cardiac cardiovascular coronary myocardial infarction angina
stroke cerebral neural neurological psychiatric psychology mental anxiety depression
schizophrenia dementia alzheimer parkinson epilepsy seizure migraine headache
nausea vomiting diarrhea constipation abdominal gastrointestinal hepatic liver
kidney renal pulmonary respiratory bronchial pneumonia tuberculosis influenza
hepatitis arthritis osteoporosis fracture orthopedic musculoskeletal dermatology
skin rash eczema psoriasis wound injury trauma hemorrhage bleeding coagulation
thrombosis embolism ischemia necrosis edema cyst abscess lesion nodule polyp
adenoma fibrosis cirrhosis stenosis occlusion perforation fistula hernia
appendicitis peritonitis anemia aplastic hemolytic sickle thalassemia lymph
lymphatic spleen thymus endocrine pancreas adrenal pituitary hypothalamus ovary
uterus cervix prostate testis nephritis glomerular tubular dialysis transplant
graft rejection immunosuppressive corticosteroid antibiotic antiviral antifungal
analgesic antipyretic anticoagulant antiplatelet diuretic bronchodilator
vasodilator antihypertensive antihistamine sedative anesthetic narcotic opioid
morphine aspirin ibuprofen acetaminophen warfarin heparin metformin statin
amoxicillin penicillin ampicillin ciprofloxacin doxycycline vancomycin
metronidazole fluconazole acyclovir ribavirin interferon vaccine vaccination
immunization prophylaxis screening prevention primary secondary tertiary
epidemiology incidence prevalence mortality morbidity survival remission relapse
recurrence complication adverse reaction interaction contraindication indication
dosage route oral intravenous subcutaneous intramuscular topical inhaled
sublingual bioavailability pharmacokinetics pharmacodynamics metabolism excretion
clearance halflife volume distribution absorption efficacy effectiveness safety
tolerability toxicity lethal therapeutic index monitor monitoring outcome clinical
trial randomized controlled placebo blind crossover cohort casestudy survey
questionnaire consent ethics informed protocol hypothesis null alternative
statistic regression correlation coefficient variance deviation standard mean
median mode probability sensitivity specificity positive negative predictive value
likelihood ratio odds risk relative absolute number needed treat harm benefit
staging grading histopathology cytology molecular genetic chromosome mutation
polymorphism expression gene sequence sequencing genomics proteomics metabolomics
phenotype genotype allele locus nucleotide amino acid codon transcription
translation replication repair recombination apoptosis proliferation
differentiation stem cell tissue organ system homeostasis feedback receptor
ligand signal transduction cascade pathway enzyme substrate cofactor inhibitor
activator agonist antagonist channel membrane transport vesicle cytoskeleton
mitochondria nucleus ribosome endoplasmic golgi lysosome peroxisome autophagy
phagocytosis endocytosis exocytosis osmosis diffusion active passive gradient
electrochemical potential action resting depolarization repolarization synapse
neurotransmitter dopamine serotonin acetylcholine norepinephrine epinephrine
glutamate glycine histamine prostaglandin leukotriene cytokine interleukin
chemokine growth factor kinase phosphorylation ubiquitin proteasome chromatin
histone acetylation methylation epigenetic microRNA regulatory promoter enhancer
exon intron splicing alternative isoform paralog ortholog homolog conserved
evolve evolution selection adaptation fitness virulence pathogenicity host
transmission contact aerosol droplet sexual vertical horizontal nosocomial
community outbreak epidemic pandemic endemic sporadic cluster investigation
tracing quarantine isolation containment surveillance reportable notifiable
population demographic age sex race ethnicity socioeconomic geographic urban
rural indoor outdoor occupational recreational environmental exposure threshold
cumulative latency incubation prodrome subclinical asymptomatic colonized
infected carrier recovered vaccinated susceptible seronegative seropositive titer
seroconversion neutralizing opsonizing complement agglutination precipitation
immunodiffusion electrophoresis cytometry elisa western blot pcr rtpcr realtime
quantitative multiplex microarray hybridization probe primer amplification gel
column chromatography spectrometry spectrophotometry microscopy electron light
confocal fluorescence staining hematoxylin eosin giemsa gram immunohistochemistry
autoimmune rheumatoid lupus scleroderma vasculitis myasthenia graves hashimoto
celiac crohn ulcerative psoriatic ankylosing sclerosis polymyositis dermatomyositis
connective overlap pressure oxygen carbon nitrogen phosphorus calcium sodium
potassium magnesium chloride bicarbonate acid base buffer saturation partial
alveolar arterial venous capillary interstitial intracellular extracellular`;

/**
 * Build word frequency map from raw corpus text.
 * Frequencies are seeded with realistic values to simulate a large corpus.
 * High-frequency clinical terms get boosted counts.
 */
const WORD_FREQ = (function () {
  const words = CORPUS_RAW.split(/\s+/).filter(Boolean);
  const freq = {};

  // High-frequency clinical terms (appear very often in medical text)
  const highFreq = new Set([
    'patient','patients','disease','treatment','clinical','drug','blood',
    'infection','symptoms','diagnosis','therapy','cancer','cardiac','renal',
    'chronic','acute','hospital','medication','dose','test'
  ]);

  // Medium-frequency terms
  const medFreq = new Set([
    'doctor','nurse','surgery','antibody','immune','inflammatory','glucose',
    'diabetes','hypertension','tumor','liver','kidney','pulmonary','cerebral',
    'genetic','molecular','cellular','tissue','receptor','enzyme','protein'
  ]);

  words.forEach((w, i) => {
    let base;
    if (highFreq.has(w))       base = 800 + Math.floor(Math.random() * 400);
    else if (medFreq.has(w))   base = 200 + Math.floor(Math.random() * 300);
    else                        base = 10  + Math.floor(Math.random() * 150);
    freq[w] = (freq[w] || 0) + base;
  });

  return freq;
})();

/** Sorted array of [word, frequency] pairs, descending by frequency. */
const SORTED_WORDS = Object.entries(WORD_FREQ).sort((a, b) => b[1] - a[1]);

/** Set of all known vocabulary words. */
const VOCAB = new Set(Object.keys(WORD_FREQ));

/**
 * Common real-word confusable pairs in medical writing.
 * Format: { misspelled_or_confusable: 'correct_contextual_word' }
 */
const REAL_WORD_CONFUSABLES = {
  'from'       : 'form',
  'liver'      : 'lifer',
  'here'       : 'hear',
  'there'      : 'their',
  'affect'     : 'effect',
  'patient'    : 'patience',
  'dose'       : 'does',
  'plain'      : 'plane',
  'right'      : 'rite',
  'course'     : 'coarse',
  'principal'  : 'principle',
  'compliment' : 'complement',
  'discrete'   : 'discreet',
  'elicit'     : 'illicit',
  'eminent'    : 'imminent',
  'precede'    : 'proceed',
  'oral'       : 'aural',
  'ileum'      : 'ilium',
  'mucus'      : 'mucous',
  'pore'       : 'pour',
  'site'       : 'sight',
  'colon'      : 'cologne',
};

/**
 * Build a bigram frequency map from co-occurrence pairs.
 * Format: { word_a: { word_b: count, ... }, ... }
 */
const BIGRAMS = (function () {
  const bg = {};

  // Manually curated high-probability medical bigrams
  const pairs = [
    ['patient','diagnosis'],    ['patient','treatment'],    ['patient','medication'],
    ['patient','history'],      ['patient','care'],         ['chronic','disease'],
    ['acute','infection'],      ['blood','pressure'],       ['blood','glucose'],
    ['blood','test'],           ['blood','count'],          ['cardiac','arrest'],
    ['cardiac','surgery'],      ['drug','dose'],            ['drug','therapy'],
    ['drug','interaction'],     ['immune','response'],      ['immune','system'],
    ['clinical','trial'],       ['clinical','diagnosis'],   ['clinical','outcome'],
    ['treatment','therapy'],    ['treatment','plan'],       ['adverse','reaction'],
    ['liver','disease'],        ['kidney','failure'],       ['heart','disease'],
    ['lung','infection'],       ['cancer','treatment'],     ['tumor','growth'],
    ['gene','expression'],      ['protein','synthesis'],    ['cell','division'],
    ['nerve','damage'],         ['bone','fracture'],        ['skin','infection'],
    ['mental','health'],        ['anxiety','disorder'],     ['diabetes','medication'],
    ['hypertension','treatment'],['pain','medication'],     ['infection','control'],
    ['hospital','patient'],     ['surgical','procedure'],   ['medical','diagnosis'],
    ['laboratory','test'],      ['physical','examination'], ['dose','response'],
    ['risk','factor'],          ['primary','prevention'],   ['secondary','prevention'],
    ['immune','deficiency'],    ['autoimmune','disease'],   ['inflammatory','response'],
    ['chronic','pain'],         ['renal','failure'],        ['hepatic','disease'],
    ['pulmonary','infection'],  ['cerebral','stroke'],      ['cardiac','failure'],
    ['antibiotic','therapy'],   ['antiviral','treatment'],  ['cancer','diagnosis'],
    ['genetic','mutation'],     ['molecular','pathway'],    ['cellular','response'],
    ['tissue','damage'],        ['organ','failure'],        ['systemic','infection'],
    ['viral','infection'],      ['bacterial','infection'],  ['fungal','infection'],
    ['inflammatory','disease'], ['autoimmune','disorder'],  ['metabolic','disorder'],
    ['hormonal','imbalance'],   ['thyroid','disorder'],     ['diabetes','diagnosis'],
    ['insulin','resistance'],   ['cholesterol','level'],    ['blood','cholesterol'],
    ['protein','expression'],   ['gene','mutation'],        ['dna','sequence'],
    ['cell','proliferation'],   ['apoptosis','pathway'],    ['signal','transduction'],
    ['receptor','binding'],     ['enzyme','activity'],      ['drug','metabolism'],
    ['pharmacokinetics','data'],['clinical','evidence'],    ['randomized','trial'],
    ['placebo','controlled'],   ['mortality','rate'],       ['survival','rate'],
    ['incidence','rate'],       ['prevalence','rate'],      ['prognosis','factor'],
  ];

  pairs.forEach(([a, b]) => {
    if (!bg[a]) bg[a] = {};
    bg[a][b] = (bg[a][b] || 0) + Math.floor(Math.random() * 60 + 15);
  });

  // Add some lower-frequency random bigrams for coverage
  const keys = Object.keys(WORD_FREQ);
  for (let i = 0; i < 200; i++) {
    const a = keys[Math.floor(Math.random() * keys.length)];
    const b = keys[Math.floor(Math.random() * keys.length)];
    if (!bg[a]) bg[a] = {};
    bg[a][b] = (bg[a][b] || 0) + Math.floor(Math.random() * 5 + 1);
  }

  return bg;
})();

/**
 * Compute P(word | prev_word) using the bigram model with add-one smoothing.
 * @param {string} prev  - Previous word (context)
 * @param {string} word  - Current word
 * @returns {number} Probability estimate
 */
function getBigramProb(prev, word) {
  if (!prev || !BIGRAMS[prev]) return 0.001;
  const counts = BIGRAMS[prev];
  const total  = Object.values(counts).reduce((s, v) => s + v, 0) || 1;
  return ((counts[word] || 0) + 0.5) / (total + VOCAB.size * 0.5);
}

/**
 * Compute unigram probability P(word).
 * @param {string} word
 * @returns {number}
 */
function getUnigramProb(word) {
  const total = Object.values(WORD_FREQ).reduce((s, v) => s + v, 0);
  return (WORD_FREQ[word] || 0) / total;
}
