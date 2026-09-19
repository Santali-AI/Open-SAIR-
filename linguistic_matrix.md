# Open SAIR: Linguistic & Orthographic Matrix Specification

This document details the exact linguistic, phonological, morphological, and orthographic matrices for the Santali language in its native Ol Chiki script, alongside its alignment with Bengali and English.

---

## 1. The 30-Letter Ol Chiki Structural Matrix

Invented by **Pandit Raghunath Murmu** in 1925, Ol Chiki is an alphabetic (non-abugida) script consisting of 30 core letters arranged in a 6×5 matrix corresponding to basic physical shapes and articulatory gestures.

| Row / Type | 1. LA (ᱚ) Series | 2. AT (ᱛ) Series | 3. AG (ᱜ) Series | 4. ANG (ᱝ) Series | 5. AL (ᱞ) Series |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vowels (Jiwi Arang)** | ᱚ (`U+1C5A`) /ɔ/ | ᱟ (`U+1C5F`) /a/ | ᱤ (`U+1C64`) /i/ | ᱩ (`U+1C69`) /u/ | ᱮ (`U+1C6E`) /e/ |
| **Checked Consonants (Keched Arang)** | ᱛ (`U+1C5B`) /t/ | ᱠ (`U+1C60`) /k/ | ᱥ (`U+1C65`) /s/ | ᱪ (`U+1C6A`) /c/ | ᱯ (`U+1C6F`) /p/ |
| **Glottal Stops (Tapug Arang)** | ᱜ (`U+1C5C`) /k̚ˀ/ | ᱡ (`U+1C61`) /c̚ˀ/ | ᱦ (`U+1C66`) /h/ | ᱫ (`U+1C6B`) /t̚ˀ/ | ᱰ (`U+1C70`) /ɖ/ |
| **Nasals (Ror Arang)** | ᱝ (`U+1C5D`) /ŋ/ | ᱢ (`U+1C62`) /m/ | ᱧ (`U+1C67`) /ɲ/ | ᱬ (`U+1C6C`) /ɳ/ | ᱱ (`U+1C71`) /n/ |
| **Liquids & Continuants (Jat Arang)** | ᱞ (`U+1C5E`) /l/ | ᱣ (`U+1C63`) /w/ | ᱨ (`U+1C68`) /r/ | ᱭ (`U+1C6D`) /j/ | ᱲ (`U+1C72`) /ɽ/ |

*Sixth Column (Row 6):*
- Vowel: ᱳ (`U+1C73`) /o/
- Plosive: ᱴ (`U+1C74`) /ʈ/
- Checked stop: ᱵ (`U+1C75`) /p̚ˀ/
- Nasal glide: ᱶ (`U+1C76`) /w̃/
- Aspiration: ᱷ (`U+1C77`) /ʰ/

---

## 2. Modifiers and Diacritics Matrix (U+1C78 to U+1C7D)

| Symbol | Unicode | Native Name | Phonetic Function | Example | Pronunciation Shift |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ᱸ** | `U+1C78` | Mu Ttuddag | Nasalization of vowel or resonant | ᱚᱸ | /ɔ/ $\rightarrow$ /ɔ̃/ |
| **ᱹ** | `U+1C79` | Gaahlaa Ttuddag | Lowers vowel height / more open articulation | ᱟᱹ | /a/ $\rightarrow$ /ə/ or open /æ/ |
| **ᱺ** | `U+1C7A` | Mu-Gaahlaa Ttuddag | Simultaneous vowel lowering + nasalization | ᱟᱺ | /a/ $\rightarrow$ /ə̃/ |
| **ᱻ** | `U+1C7B` | Relaa | Vowel prolongation / lengthening | ᱮᱻ | /e/ $\rightarrow$ /eː/ |
| **ᱼ** | `U+1C7C` | Phaarkaa | Glottal separator (preserves checked stop from assimilation) | ᱫᱼ | Prevents sandhi across syllable boundary |
| **ᱽ** | `U+1C7D` | Ahad | Deglottalizer: softens unreleased stop to voiced plosive | ᱜᱽ, ᱡᱽ, ᱫᱽ, ᱵᱽ | /k̚ˀ/ $\rightarrow$ /g/, /c̚ˀ/ $\rightarrow$ /ɟ/, /t̚ˀ/ $\rightarrow$ /d/, /p̚ˀ/ $\rightarrow$ /b/ |
| **᱾** | `U+1C7E` | Mucaad | Sentence terminator (Single Danda) | ᱾ | Full stop equivalent |
| **᱿** | `U+1C7F` | Double Mucaad | Paragraph/stanza terminator (Double Danda) | ᱿ | Poetic stanza / paragraph end |

---

## 3. The Santali Pronoun Matrix Engine

Santali exhibits a rich three-way grammatical number system (**Singular**, **Dual**, **Plural**) and an essential distinction in the first person between **Inclusive** (listener included) and **Exclusive** (listener excluded).

| Person | Number | Clusivity | Animacy / Deixis | Ol Chiki | Latin | Subj Clitic | Obj Clitic | English Equivalent | Bengali Equivalent |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1st** | Singular | N/A | Animate | **ᱤᱧ** | iń | -ᱧ (-ń) | -ᱧ (-ń) | I, me | আমি, আমাকে |
| **1st** | Dual | **Inclusive** | Animate | **ᱟᱞᱟᱝ** | alaṅ | -ᱞᱟᱝ (-laṅ) | -ᱞᱟᱝ (-laṅ) | We two (You and I) | আমরা দুজন (তুমি ও আমি) |
| **1st** | Dual | **Exclusive** | Animate | **ᱟᱞᱤᱧ** | aliń | -ᱞᱤᱧ (-liń) | -ᱞᱤᱧ (-liń) | We two (He/She and I, not you) | আমরা দুজন (সে ও আমি, তুমি নও) |
| **1st** | Plural | **Inclusive** | Animate | **ᱟᱵᱚ** / **ᱟᱵᱚᱱ** | abo / abon | -ᱵᱚ (-bo) | -ᱵᱚ (-bo) | We all (All of us including you) | আমরা সবাই (তোমারা সহ) |
| **1st** | Plural | **Exclusive** | Animate | **ᱟᱞᱮ** | ale | -ᱞᱮ (-le) | -ᱞᱮ (-le) | We all (All of us excluding you) | আমরা (তোমরা ছাড়া) |
| **2nd** | Singular | N/A | Animate | **ᱟᱢ** | am | -ᱢ (-m) | -ᱢᱮ (-me) | You (singular) | তুমি, তুই, আপনি |
| **2nd** | Dual | N/A | Animate | **ᱟᱵᱮᱱ** | aben | -ᱵᱮᱱ (-ben) | -ᱵᱮᱱ (-ben) | You two | তোমরা দুজন, আপনারা দুজন |
| **2nd** | Plural | N/A | Animate | **ᱟᱯᱮ** | ape | -ᱯᱮ (-pe) | -ᱯᱮ (-pe) | You all | তোমরা, আপনারা |
| **3rd** | Singular | N/A | Animate (Dist) | **ᱩᱱᱤ** | uni | -ᱭ (-y) | -ᱮ (-e) | He / She (that person) | সে, তিনি, তাকে |
| **3rd** | Singular | N/A | Animate (Prox) | **ᱱᱩᱭ** | nui | -ᱭ (-y) | -ᱮ (-e) | This person (he/she here) | এ, ইনি, একে |
| **3rd** | Dual | N/A | Animate (Dist) | **ᱩᱱᱠᱤᱱ** | unkin | -ᱠᱤᱱ (-kin) | -ᱠᱤᱱ (-kin) | They two (those two) | তারা দুজন, ওঁরা দুজন |
| **3rd** | Dual | N/A | Animate (Prox) | **ᱱᱩᱠᱤᱱ** | nukin | -ᱠᱤᱱ (-kin) | -ᱠᱤᱱ (-kin) | These two | এরা দুজন |
| **3rd** | Plural | N/A | Animate (Dist) | **ᱩᱱᱠᱩ** | unku | -ᱠᱚ (-ko) | -ᱠᱚ (-ko) | They all, them | তারা, ওঁরা, তাদের |
| **3rd** | Plural | N/A | Animate (Prox) | **ᱱᱩᱠᱩ** | nuku | -ᱠᱚ (-ko) | -ᱠᱚ (-ko) | These people | এরা, এদের |
| **3rd** | Singular | N/A | Inanimate (Dist)| **ᱚᱱᱟ** | ona | $\emptyset$ | $\emptyset$ | That (thing) | ওটা, সেটি |
| **3rd** | Singular | N/A | Inanimate (Prox)| **ᱱᱚᱣᱟ** | nowa | $\emptyset$ | $\emptyset$ | This (thing) | এটা, এটি |

---

## 4. Agglutinative Verb Morphology and TAM Infixation

Santali predicates are polysynthetic agglutinations following the structural template:
$$\text{Predicate} = \text{Root} + [\text{Reciprocal } -p-] + [\text{Tense/Aspect}] + [\text{Voice}] + [\text{Object Clitic}] + [\text{Finite } -a] + [\text{Subject Clitic}]$$

### 4.1. Reciprocal Infixation ($-p-$ / ᱯ)
The reciprocal marker $-p-$ is infixed directly inside the verbal root immediately after the first vowel, followed by an echo of that vowel:
- Root: **ᱫᱟᱞ** (*dal*, "hit") $\rightarrow$ Infix: **ᱫᱟᱯᱟᱞ** (*d-ap-al*, "hit each other / fight")
- Root: **ᱧᱮᱞ** (*ñel*, "see") $\rightarrow$ Infix: **ᱧᱮᱯᱮᱞ** (*ñ-ep-el*, "meet / see each other")
- Root: **ᱜᱚᱪ** (*goc'*, "kill") $\rightarrow$ Infix: **ᱜᱚᱯᱚᱪ** (*g-op-oc'*, "kill each other")

### 4.2. Tense-Aspect-Mood (TAM) Markers

| TAM Category | Ol Chiki Form | Latin | Transitivity | Example Predicate | Gloss Breakdown | English Meaning |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Present Cont.** | -ᱠᱟᱱ- | -kan- | Neutral | ᱫᱟᱞ-ᱮᱫ-ᱮ-**ᱠᱟᱱ**-ᱟ-ᱭ | hit-TRANS-3SG.OBJ-**CONT**-FIN-3SG.SUBJ | He is hitting him |
| **Simple Past** | -ᱠᱮᱫ- | -ked- | Transitive | ᱫᱟᱞ-**ᱠᱮᱫ**-ᱮ-ᱟ-ᱧ | hit-**PAST.TRANS**-3SG.OBJ-FIN-1SG.SUBJ | I hit him |
| **Past Intrans.**| -ᱮᱱ- | -en- | Intransitive | ᱪᱟᱞᱟᱣ-**ᱮᱱ**-ᱟ-ᱧ | go-**PAST.INTRANS**-FIN-1SG.SUBJ | I went |
| **Perfect** | -ᱟᱠᱟᱫ- | -akad- | Transitive | ᱡᱚᱢ-**ᱟᱠᱟᱫ**-ᱟ-ᱧ | eat-**PERF.TRANS**-FIN-1SG.SUBJ | I have eaten |
| **Past Perfect** | -ᱞᱮᱫ- | -led- | Transitive | ᱧᱮᱞ-**ᱞᱮᱫ**-ᱮ-ᱟ-ᱧ | see-**PLUPERF.TRANS**-3SG.OBJ-FIN-1SG.SUBJ | I had seen him |

---

## 5. Cross-Lingual Word Order and Postpositional Transformation

| Dimension | English | Bengali | Santali |
| :--- | :--- | :--- | :--- |
| **Syntactic Typology** | **SVO** (Analytic) | **SOV** (Inflectional) | **SOV** (Polysynthetic Agglutinative) |
| **Adposition Typology** | Prepositional (in, to, from) | Postpositional (-এ, -তে, থেকে) | Postpositional / Suffixal (-ᱨᱮ, -ᱛᱮ, -ᱠᱷᱚᱱ) |
| **Locative Marker** | in the village | গ্রামে (gram-e) | ᱟᱹᱛᱩ**ᱨᱮ** (atu-**re**) |
| **Instrumental Marker** | with an axe | কুড়াল দিয়ে (kural diye) | ᱴᱟᱸᱜᱟ**ᱛᱮ** (ṭaṅga-**te**) |
| **Ablative Marker** | from the house | বাড়ি থেকে (bari theke) | ᱚᱲᱟᱜ**ᱠᱷᱚᱱ** (oṛak'-**khon**) |
| **Animate Genitive** | the man's son | লোকটির ছেলে (lok-tir chele) | ᱦᱚᱲ**ᱨᱮᱱ** ᱦᱚᱯᱚᱱ (hoṛ-**ren** hopon) |
| **Inanimate Genitive** | door of the house | ঘরের দরজা (ghor-er dorja) | ᱚᱲᱟᱜ**ᱟᱜ** ᱫᱩᱣᱟᱹᱨ (oṛak'-**ak'** duwạr) |
