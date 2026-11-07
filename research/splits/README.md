# Team train / val / test splits

These are the group's own splits for EDF, MCFD and UP-Fall. Every result in `../results/` uses them.
They are **not** the splits published with OmniFall (`/scratch/wuhungma/omnifall/splits/`); see the bottom of this file.

**Training or evaluating something new? Use `frames32_{train,val,test}.jsonl`.** It holds all three datasets with frame paths.

## Which file is which

| file | what it is | rows (train / val / test) |
|---|---|---|
| `frames32_{train,val,test}.jsonl` | **Use this.** All three datasets, one row per labelled event, with paths to its 32 frames. Exactly `{split}.jsonl` + `up_fall_events_{split}.jsonl`. | 1151 / 209 / 274 |
| `{train,val,test}.jsonl` | Split definition for **EDF + MCFD** (one row per labelled event) | 292 / 67 / 64 |
| `up_fall_{train,val,test}.jsonl` | Split definition for **UP-Fall** (one row per recording / trial) | 396 / 66 / 99 |
| `up_fall_events_{train,val,test}.jsonl` | UP-Fall recordings cut into labelled events with OmniFall's timings. Same people as `up_fall_*.jsonl`. | 859 / 142 / 210 |
| `up_fall_frame_index.jsonl` | Bookkeeping for `../data/extract_upfall_frames.py` (where each recording's frames went). Not a split. | 561 total |
| `old/` | Outdated files kept for reference. Not used by anything. | |

## Who is in which split

| dataset | split unit | train | val | test |
|---|---|---|---|---|
| EDF | person | jianjun, peter, zhong | jinhui | songsong |
| MCFD | chute (scenario); one actor throughout | the other 16 chutes | chute02, 05, 17, 21 | chute01, 03, 13, 18 |
| UP-Fall | person | subjects 2, 3, 6-11, 13-16 | 5, 17 | 1, 4, 12 |

No person (EDF, UP-Fall) or chute (MCFD) appears in more than one split. MCFD has a single actor, so its
test set is the same person in the same room; MCFD numbers are optimistic for any trained model.

## How the files were made

- `../data/split_dataset.py`: EDF + MCFD, whole scenarios per split, about 70 / 15 / 15
- `../data/split_upfall.py`: UP-Fall, whole subjects per split (12 / 2 / 3 people)
- `../data/build_upfall_events.py`: `up_fall_*` -> `up_fall_events_*`
- `../data/make_frames_manifest.py`: -> `frames32_*`

## Difference from OmniFall's published splits

OmniFall's own split CSVs are in `/scratch/wuhungma/omnifall/splits/{cs,cv}/<dataset>/`. They assign different
people to test, so our numbers are not comparable with published OmniFall numbers. They also have problems:
UP-Fall cross-subject has subjects 15 and 16 in both train and test; MCFD cross-subject has empty val and test;
some cross-view files are duplicated (EDF val = test, UP-Fall train = val).
Each camera row in `{train,val,test}.jsonl` keeps OmniFall's assignment in `split_cs` / `split_cv`.
