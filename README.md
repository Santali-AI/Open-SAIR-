# Santali–Bengali–English Trilingual Vocabulary

A GitHub-ready working vocabulary resource in the format:

**Santali (Ol Chiki) → Bengali → English**

## Dataset

- File: `santali_bengali_english_vocabulary_1000.csv`
- Rows: 1000
- Script: Ol Chiki (Unicode)
- Columns: `id`, `category`, `santali_ol_chiki`, `bengali`, `english`, `status`, `notes`

## Important quality note

This repository intentionally distinguishes **seeded/source-backed entries** from **review candidates**. It does not fabricate Santali spellings merely to reach 1000 rows.

Before treating the dataset as a finished linguistic resource, have the `needs_native_review` rows checked by fluent/native Santali speakers, preferably covering West Bengal/Jharkhand/Odisha varieties as appropriate.

## Suggested GitHub workflow

1. Add one verified vocabulary item per row.
2. Keep Ol Chiki spelling in Unicode.
3. Record regional variants in a separate `variant` column when needed.
4. Keep part of speech/category consistent.
5. Use pull requests for corrections.
6. Add a `source` column for every verified entry.
7. Do not silently replace regional variants; preserve them as variants.

## Sources consulted

- CIIL/NCERT Santali–Bengali Primer (2024).
- LanguageKnow Santali dictionary index.
- Kaikki machine-readable Santali/Wiktionary-derived dictionary.
- A. Campbell, *A Santali-English Dictionary* (1899, public domain).
- R. C. Hansdah & N. C. Murmu, *A Concise Santali-English Dictionary* (2003).
- Bharatavani/CIIL Santali dictionary resources.

## License recommendation

For the **new data you personally verify and contribute**, choose a license suitable for your repository (for example CC BY 4.0 for data, or CC0 if you want maximum reuse). Do not assume that every upstream dictionary has the same license.
