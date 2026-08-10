# Yandex Cup 2023 "NeuroSwipe" corpus — licence / usage-terms research

Date: 2026-08-10 · Scope: web-only investigation, no contact with any party.
Subject: the ~6 M ЙЦУКЕН swipe corpus used in the Yandex Cup 2023 ML track,
downloaded to `~/ctc-train/data/yandex_cup/` via the Yandex Disk public API.

> **Not legal advice.** This is an engineering risk memo assembled from primary
> sources. Every claim below is quoted with its URL so the owner can re-check.

---

## 0. Verdict up front

The working assumption **"open to use unless stated otherwise" does not hold.**

There is **no licence grant of any kind** attached to this corpus — not on the
contest page, not in the Yandex Cup regulations, not on the Yandex Disk link,
not in any solution repo, not on the Kaggle mirror ("License: Unknown"), and
there is no dataset card or paper releasing it. Absence of a grant is not
permission: the corpus is a Russian-law **database whose maker (ООО «Яндекс»)
holds a statutory exclusive right to extract and reuse its contents** (ГК РФ
ст. 1334), a right that runs to ~2039 and exists *independently* of copyright in
the individual records. The statutory carve-out that lets a lawful user extract
freely (ст. 1335.1) is limited to *the purpose the database was provided for*
(the competition), plus *personal / scientific / educational* use, plus
*insubstantial parts* for anything else. Training a shipped keyboard model on
6 M rows is none of those.

Recommendation: **(b) research/eval-only for the Yandex corpus, (c) synth-only
for anything that ships** — with the Yandex data demoted to a held-out
validation set. That is precisely what FUTO did in the one published precedent.

---

## 1. What we actually hold

`~/ctc-train/data/yandex_cup/` (19 GB unpacked), from
`https://disk.yandex.ru/d/IYiSpLob-zAxqg` (`data.zip`, 1,745,670,429 B,
sha256 `2e65d7a2…b37521`, matching the Disk API checksum):

| file | content |
|---|---|
| `train.jsonl` | 6,000,000 curves + target words (17.6 GB) |
| `valid.jsonl` + `valid.ref` | 10,000 curves + targets |
| `test.jsonl` | curves, no targets |
| `voc.txt` | 503,598 words |

We did **not** download the *second* archive offered by the task
(`https://disk.yandex.ru/d/-qAoI9Ux1eP7XQ`, the `accepted` /
`suggestion_accepted` sets). That distinction matters — see §5.

---

## 2. Explicit terms: exhaustive search, nothing found

### 2.1 The contest task statement — describes the data, grants nothing

Source: the problem statement mirrored verbatim in the 7th-place solution repo
(`contest.yandex.ru/contest/54253/problems` itself is login-walled and has no
Wayback capture of the problem body):
<https://github.com/proshian/neural-swipe-typing/blob/main/docs_and_assets/yandex_cup/task/task.md>

Provenance of the main sets, verbatim:

> «Данные были собраны путем разметки, когда пользователей просили ввести
> слово, отображенное на экране, с использованием свайпа.»
> *(The data were collected by annotation, where users were asked to enter a
> word displayed on the screen using a swipe.)*

Provenance of the *additional* archive, verbatim:

> «Так же мы предоставляем дополнительный архив с кривыми, которые вводились
> пользователями в Яндекс.Клавиатуре»
> *(We also provide an additional archive with curves that were entered by users
> in Yandex.Keyboard.)*

Download instruction, verbatim: «Скачать данные можно по [ссылке].»

That is the entire treatment of the data. **No licence, no permitted-use
clause, no restriction, no attribution requirement, no post-contest statement.**

### 2.2 Yandex Cup 2023 «Положение о конкурсе» (regulations) — no data clause

2023-era snapshot:
<https://web.archive.org/web/20231027143336/https://yandex.ru/cup/regulations/>
(current: <https://yandex.ru/cup/regulations>)

Grepped for `данны|материал|лиценз|датасет|предоставля`. The only IP clause runs
the *opposite* direction — participant → organiser (cl. 6.8, verbatim):

> «Направляя Результаты Организатору, Участник … сохраняет все права на
> интеллектуальную собственность в отношении своих Результатов, но
> предоставляет Организатору безвозмездную неисключительную (простую) лицензию
> в отношении права использовать такие Результаты…»

English equivalent, <https://yandex.com/cup/regulations>:

> "By sending Results to the Organizer, the Participant retains all intellectual
> property rights in relation to their Results, but grants the Organizer a
> royalty-free non-exclusive license to use the Results…"

The only other restriction is cl. 5.3, about not sharing *solutions*:

> «Участники обязуются не обсуждать Задания и их решение с другими Участниками и
> третьими лицами… Запрещается публиковать решения Заданий в сети Интернет…»

**Nothing anywhere about the organiser's data.**

### 2.3 Yandex Cup 2023 «Правила проведения» (rules) — no data clause

<https://web.archive.org/web/20231027143323/https://yandex.ru/cup/rules/>
(current <https://yandex.ru/cup/rules>). Same grep. Every hit for `данны` /
`датасет` is about the judging system (input/output formats, private test
dataset re-scoring). No usage terms for the distributed corpus.

### 2.4 «Общие условия проведения конкурсов» — participant→organiser only

<https://yandex.ru/legal/competition_generalterms/> (incorporated by reference
via reg. cl. 6.5). Clauses 4.4–4.7 govern *РИД* = intellectual-activity results
**supplied by the participant**; cl. 4.6 even assigns winners' exclusive rights
to the organiser. There is no reciprocal clause governing materials the
organiser supplies to participants.

### 2.5 Third-party mirrors and solution repos — no licence

| source | licence |
|---|---|
| Kaggle mirror `sharthz23/yandex-cup-2023-neuroswipe` (5.18 GB, incl. the `accepted` sets) | **"License: Unknown"** — verbatim field on <https://www.kaggle.com/datasets/sharthz23/yandex-cup-2023-neuroswipe> |
| `proshian/neural-swipe-typing` (7th place, the repo that documents the Disk link) | GitHub API `license: None`; no `LICENSE` file (HTTP 404) |
| `kbrodt/yandex-cup-2023-neuroswipe` (1st place) | GitHub API `license: None` |
| other `neuroswipe` repos | mostly none; two unrelated forks are MIT-licensed **as to their own code**, which says nothing about the data |

A third-party uploader tagging a Kaggle mirror cannot confer rights they do not
hold, and they did not even claim to — the field reads *Unknown*.

### 2.6 No dataset card, no paper, no HF release

- HuggingFace API search for `neuroswipe` (datasets **and** models): empty.
- No arXiv/academic paper by Yandex releasing or documenting this corpus.
- The only academic use found is *third-party* (§6).

**Conclusion of §2: the dataset was published with zero stated terms.**

---

## 3. What the distribution channel implies (the terms that *do* apply)

### 3.1 Yandex's blanket user agreement covers content obtained via Yandex services

The corpus was distributed exclusively through Yandex services (contest.yandex.ru
→ a `disk.yandex.ru` public link → `cloud-api.yandex.net` download endpoint).
The «Пользовательское соглашение сервисов Яндекса» therefore forms the
background terms.

<https://yandex.com/legal/rules/> (English), cl. 2.8.1 verbatim:

> "The User may not reproduce, duplicate or copy, sell, resell or **use for any
> commercial purposes any parts of Yandex services (including content available
> to the User through services)** or access to Yandex services, except when
> authorized by Yandex or it is directly stated in the user agreement for any
> service."

cl. 6.2 verbatim:

> "Any content and service elements may be used **only within functions offered
> by a particular service**. No elements of Yandex service content as well as
> any content posted at Yandex services **may be used in any other way without
> the right holder's prior consent**. The term 'use' shall include reproduction,
> duplication, processing and distribution on any basis… **The personal
> non-commercial use** by the User of service content elements and any content
> **is authorized** upon preservation of all marks of copyright…"

Russian original, <https://yandex.ru/legal/rules/> cl. 5.2.10 / 6.2, same effect:

> «…воспроизводить, повторять и копировать, продавать и перепродавать, а также
> использовать для каких-либо коммерческих целей какие-либо части сервисов
> Яндекса (включая контент, доступный Пользователю посредством сервисов
> Яндекса)… кроме тех случаев, когда Пользователь получил такое разрешение от
> Яндекса»
>
> «Использование Пользователем элементов Содержания сервисов Яндекса… для
> личного некоммерческого использования, допускается при условии сохранения
> всех знаков охраны авторского права…»

So the *default* permission Yandex extends over content reached through its
services is **personal, non-commercial, unmodified, attributed** — the opposite
end of the spectrum from "open unless stated otherwise".

### 3.2 Yandex Disk terms grant recipients nothing

<https://yandex.ru/legal/disk_termsofuse/> cl. 2.17 describes public links purely
as a *distribution mechanism* available to the uploader:

> «Диск предоставляет Пользователю возможность распространять данные путём…
> предоставления публичного доступа к файлу (предоставление доступа к файлу
> пользователям сети Интернет, которым известен адрес файла в виде ссылки).»

Publishing a public link makes a file *reachable*. It is not a licence, and the
Disk terms contain no clause granting downloaders any rights.

### 3.3 The decisive instrument: Russian database-maker's right (ГК РФ §5 гл. 71)

This is the exposure that survives even if one argues the raw coordinates are
uncopyrightable facts. Russia (like the EU, unlike the US) has a *sui generis*
database right.

**ст. 1333** — the maker is «лицо, организовавшее создание базы данных и работу
по сбору, обработке и расположению составляющих ее материалов». That is ООО
«Яндекс»: it organised the elicitation campaign, the keyboard instrumentation and
the packaging. <https://www.zakonrf.info/gk/1333/>

**ст. 1334 п.1** verbatim (<https://www.zakonrf.info/gk/1334/>):

> «Изготовителю базы данных, создание которой… требует существенных финансовых,
> материальных, организационных или иных затрат, принадлежит исключительное
> право **извлекать из базы данных материалы и осуществлять их последующее
> использование в любой форме и любым способом**…
> При отсутствии доказательств иного базой данных, создание которой требует
> существенных затрат, признается база данных, **содержащая не менее десяти
> тысяч самостоятельных информационных элементов**…
> Никто не вправе извлекать из базы данных материалы и осуществлять их
> последующее использование без разрешения правообладателя… При этом под
> **извлечением** материалов понимается перенос всего содержания базы данных или
> **существенной части** составляющих ее материалов на другой информационный
> носитель…»

6,000,000 rows clears the 10,000-element presumption by 600×; a full-archive
download is textbook «извлечение… всего содержания». Note **ст. 1334 п.2**: the
right «признается и действует независимо от наличия и действия авторских… прав…
на составляющие базу данных материалы» — i.e. "the coordinates are mere facts"
is not a defence to *this* right.

**ст. 1335** — term is 15 years from 1 January of the year following publication,
renewed on each update. Published 2023 ⇒ protected to at least **2039**.
<https://www.zakonrf.info/gk/1335/>

**ст. 1335.1 п.1** — the exceptions, and they map startlingly well onto our exact
question (<https://www.zakonrf.info/gk/1335.1/>):

> «Лицо, правомерно пользующееся обнародованной базой данных, вправе без
> разрешения… извлекать из базы данных материалы и осуществлять их последующее
> использование:
> — **в целях, для которых база данных ему предоставлена**, в любом объеме, если
>   иное не предусмотрено договором;
> — **в личных, научных, образовательных целях** в объеме, оправданном
>   указанными целями;
> — **в иных целях в объеме, составляющем несущественную часть базы данных.**
> Использование материалов, извлеченных из базы данных, способом, предполагающим
> получение к ним доступа неограниченного круга лиц, **должно сопровождаться
> указанием на базу данных**, из которой эти материалы извлечены.»

Applied to us:

- *"the purpose for which it was provided"* = **competing in Yandex Cup 2023**.
  The contest closed in 2023 and we were not participants. This limb does not
  reach a 2026 product.
- *"personal, scientific, educational"* — a research measurement (what Phase I-B
  actually did: measure whether Cyrillic decoding is feasible, report numbers)
  sits **inside** this limb. Shipping trained weights in a distributed keyboard
  sits **outside** it.
- *"other purposes, insubstantial part only"* — training on the full 6 M rows is
  the substantial part by definition.

There is also **ст. 1335.1 п.4**: the maker cannot restrict individual materials
lawfully obtained *from other sources*. This is the legal basis for the
synthetic/independent-collection route — nothing about the Yandex right touches
data we generate or collect ourselves.

### 3.4 Precedent: Yandex licenses its intentional data releases explicitly

When Yandex Research means to release a dataset for outside use, it attaches a
licence — the Shifts dataset is **CC BY-NC-SA 4.0**
(<https://github.com/yandex-research/shifts>, <https://shifts.ai/dataset>).
Two implications: (i) the silence here is not an oversight-shaped grant, it is
simply silence; (ii) even Yandex's *deliberate* open-data posture is
**non-commercial + share-alike**, which is itself GPL-incompatible.

---

## 4. Jurisdictional nuance (why the risk is not uniform)

| forum | analysis |
|---|---|
| **Russia** | Strongest claim against us: ст. 1334 sui generis right, no applicable exception for a shipped product, term to 2039. Yandex Cup's own dispute clause points at Хамовнический суд, Moscow (reg. cl. 11.4) — though that binds *registered participants*, which we are not. |
| **EU** | Parallel sui generis right (Database Directive). DSM art. 4 permits commercial TDM **unless rights were expressly reserved in a machine-readable way** — no such reservation is present on the Disk link, so an EU TDM defence is plausible for *training*; it does not obviously cover redistributing outputs, and F-Droid ships EU-wide. |
| **US** | Weakest claim against us: no sui generis database right (*Feist*); coordinates are facts; training is increasingly treated as non-infringing. But contract/ToS (§3.1) is not preempted the same way. |

Practical read: the legal exposure is real but concentrated in a jurisdiction
where a Russian corporate rightsholder would have to choose to pursue a GPL
keyboard project. The *reputational* and *distribution-channel* exposure (§7.1)
is more likely to bite than a lawsuit.

---

## 5. Provenance / privacy note

The two archives have materially different provenance, and this is one place we
are already in good shape:

- **What we have** (main archive): elicited data — annotators were shown a word
  and asked to swipe it (§2.1). No user identifiers, no free text, targets drawn
  from a fixed 503,598-word vocabulary. This is a commissioned collection, not
  intercepted typing.
- **What we deliberately do not have** (`accepted` / `suggestion_accepted`,
  `disk.yandex.ru/d/-qAoI9Ux1eP7XQ`): real production input from Яндекс.Клавиатура
  users. That set carries a genuine personal-data dimension (what real people
  actually typed) on top of the database-right question, and there is no consent
  record we can inspect. **Do not download it, and do not use the Kaggle mirror,
  which bundles it** (`accepted_curves.parquet`, 2.93 GB).

Keeping to the elicited archive removes the privacy limb of the risk and leaves
only the rights limb.

---

## 6. Precedent: how the one comparable actor handled it

FUTO — who ship a keyboard and publish their swipe work — used this exact corpus
and used it **evaluation-only**. *FUTO Swipe: Layout-Agnostic Neural Swipe
Decoding*, arXiv:2606.25247, §4.1, verbatim:

> "Russian swipe validation results come from the Yandex Cup 2023 NeuroSwipe
> data². The Yandex corpus covers two Cyrillic JCUKEN layouts: RU-A (31 keys,
> 9,416 val samples…) and RU-B (32 keys, 584 val samples). … **The encoder is
> trained on English swipe.futo.org only.**"

and, on their own release:

> "We release swipe.futo.org, **the largest MIT-licensed swipe corpus we are
> aware of**… The corpus is released under the MIT license."

They built their own MIT corpus to train on and used Yandex purely as a
held-out cross-layout probe. They do not state a reason, but the shape of the
decision is unambiguous and it is the same shape recommended here.

Note also: **How-We-Swipe contains no Russian.** Our local `metadata.tsv`
(909 users) breaks down as en 815, es 40, ar 9, ko 7, fr 7, tr 3, fi 3, zh 3,
pl 3, it 2, sv 2, de 2. There is no licensed real Cyrillic swipe corpus
available to substitute in.

---

## 7. Risk assessment

### (a) Train a shipped GPL-3.0 / F-Droid model on it — **HIGH, and structurally awkward**

1. **Rights.** Training on 6 M rows is extraction + subsequent use of a
   substantial part of a protected database, for a purpose outside every
   ст. 1335.1 carve-out. No grant exists to rely on.
2. **The GPL conflict is the cleanest argument against it.** Every permission we
   could plausibly claim is *non-commercial* — the ToS default (§3.1: "personal
   non-commercial use is authorized"), the ст. 1335.1 научные/личные limb, and
   even Yandex's own explicit dataset posture (CC BY-NC-SA). GPL-3.0 grants
   freedom 0: use for **any** purpose, commercial included, by every downstream
   recipient. **We cannot ship an artefact under GPL-3.0 while the only theory
   permitting its existence is a non-commercial one.** Anyone forking CleverKeys
   commercially would inherit a defect we knowingly created.
3. **F-Droid.** Inclusion Policy: *"All assets need to have valid legal licenses
   or be in the public domain while being free of copyright infringement"*, and
   non-commercial-licensed assets *"must allow redistribution"*
   (<https://f-droid.org/en/docs/Inclusion_Policy/>). The policy does not yet
   speak to model weights explicitly, so this would not auto-fail a build — but
   if challenged we would have no licence to point at, and the honest answer
   ("trained on an unlicensed competition corpus") invites a `NonFreeAssets`
   anti-feature argument or a removal request.
4. **Detectability / takedown.** Moderate. We would have to document the data
   source to satisfy our own reproducibility standards, which makes the exposure
   self-published. A rightsholder complaint to F-Droid is cheaper for them than
   litigation and more likely.
5. Mitigations that do **not** work: laundering through a synth-pretrain +
   real-finetune schedule (still trained on the corpus); using "only" 94 k rows
   (still a substantial part in absolute terms, and 94 k ≫ insubstantial);
   shipping weights rather than data (ст. 1334 covers «последующее использование
   в любой форме и любым способом»).

### (b) Keep it research-only (train/measure locally, never ship) — **LOW**

This is where the ст. 1335.1 «в личных, научных, образовательных целях» limb
actually lands. Concretely permitted under this reading:

- keeping the corpus on a dev machine;
- training research models on it to measure feasibility and to bound what
  Cyrillic quality is achievable;
- using `valid.jsonl`/`valid.ref` as a **held-out benchmark** for models trained
  on other data (the FUTO pattern) — this is the highest-value use and the
  lowest-risk one;
- publishing *numbers* and methodology, with attribution to the corpus (the
  ст. 1335.1 attribution sentence).

Not permitted even here: redistributing the corpus or a derived npz, committing
slices to the repo, or publishing preprocessed mirrors.

Residual risk is near zero and the research value is largely preserved: an
honest RU eval footing is most of what the corpus buys us.

### (c) Ship synth-only Cyrillic (no Yandex data in the training path) — **NEGLIGIBLE legal, REAL quality cost**

Legally clean: ст. 1335.1 п.4 explicitly protects materials obtained from other
sources, our generator's output is our own, and GPL-3.0 + F-Droid are satisfied
without caveat.

The cost is measured, not hypothetical (PHASE_I_DATA.md, this repo):

| arm | in-dict top-1 on real RU data |
|---|---|
| ru-real (94 k Yandex rows) | **89.64** |
| ru-synth (residual transplant, zero real rows) | **76.21** |

≈13 points of top-1. 76.2 is "geometric-engine class" — shippable as a first
Cyrillic release, clearly not English-class (87–89). Note the 89.64 figure was
itself measured *on the Yandex val split*, so even the comparison depends on
keeping (b) alive as an evaluation footing.

---

## 8. Recommendation

**Adopt (b) + (c) together, and record the boundary in the repo.**

1. **Nothing derived from the Yandex corpus enters a shipped artefact.** No
   weights, no distilled teacher, no synth generator fitted to Yandex residuals,
   no vocabulary lifted from `voc.txt`. Draw the line at the training pipeline
   boundary and keep it auditable.
2. **Demote the corpus to a held-out RU evaluation set.** `valid.jsonl` +
   `valid.ref` (10 k rows) as the RU benchmark, cited as
   "Yandex Cup 2023 NeuroSwipe, evaluation only". This is the FUTO pattern, it
   preserves the honest measurement, and it is the use most defensible under
   ст. 1335.1.
3. **Ship Cyrillic from the synthetic generator** if Cyrillic ships at all, and
   be explicit in release notes that RU is synth-trained and lower-accuracy than
   EN. 76.2 with a clean provenance beats 89.6 with an unlicensed one.
4. **Never download or touch the `accepted`/`suggestion_accepted` archive** or
   the Kaggle mirror that bundles it (§5).
5. **Do not redistribute** the corpus or any preprocessed derivative; keep it out
   of git (already the case — `~/ctc-train/data/yandex_cup/` is untracked).
6. **The one action that changes this analysis: ask Yandex.** A written
   permission for open-source, GPL-compatible use would move (a) from HIGH to
   LOW. Absent that, treat the file as read-only research material.
   *(Per the standing instruction, no contact was made and none should be made
   without explicit per-instance approval.)*
7. **The durable fix is our own data.** No licensed real Cyrillic swipe corpus
   exists (§6). Options, in rough order of cost: contribute Russian prompts to
   swipe.futo.org (MIT, already our main training corpus, and upstream benefits);
   stand up a donation flow of our own; or invest further in the synthetic
   generator, whose 76.2 was a first attempt with headroom.

**Bottom line for the user's assumption:** it is refuted, but gently — there is
no rule forbidding you from *having* or *studying* this data, and the research
you have already done on it is fine. What is not supportable is shipping a
GPL-3.0 model trained on it, because every permission theory available is
non-commercial and GPL is not.

---

## 9. Sources

- Task statement (mirrored): <https://github.com/proshian/neural-swipe-typing/blob/main/docs_and_assets/yandex_cup/task/task.md>
- Contest (login-walled): <https://contest.yandex.ru/contest/54253/problems/>
- Yandex Cup regulations, 2023 snapshot: <https://web.archive.org/web/20231027143336/https://yandex.ru/cup/regulations/>
- Yandex Cup rules, 2023 snapshot: <https://web.archive.org/web/20231027143323/https://yandex.ru/cup/rules/>
- Yandex Cup regulations (current, RU/EN): <https://yandex.ru/cup/regulations> · <https://yandex.com/cup/regulations>
- General competition terms: <https://yandex.ru/legal/competition_generalterms/>
- Yandex services user agreement: <https://yandex.ru/legal/rules/> · <https://yandex.com/legal/rules/>
- Yandex Disk terms of use: <https://yandex.ru/legal/disk_termsofuse/>
- ГК РФ ст. 1333 / 1334 / 1335 / 1335.1: <https://www.zakonrf.info/gk/1333/> · <https://www.zakonrf.info/gk/1334/> · <https://www.zakonrf.info/gk/1335/> · <https://www.zakonrf.info/gk/1335.1/>
- Kaggle mirror (License: Unknown): <https://www.kaggle.com/datasets/sharthz23/yandex-cup-2023-neuroswipe>
- Solution repos: <https://github.com/proshian/neural-swipe-typing> · <https://github.com/kbrodt/yandex-cup-2023-neuroswipe>
- Yandex Shifts (licensed release precedent): <https://github.com/yandex-research/shifts> · <https://shifts.ai/dataset>
- FUTO Swipe (eval-only precedent): <https://arxiv.org/html/2606.25247>
- F-Droid Inclusion Policy: <https://f-droid.org/en/docs/Inclusion_Policy/>

---

## 10. Re-review: the proshian precedent

Date: 2026-08-10 · Scope: web-only, no contact with any party. This section
**does not revise §§0–9**; it tests the eval-only recommendation against one
specific counter-argument raised after the fact:

> *proshian (7th place, Yandex Cup 2023) trained on this corpus and published
> models. Doesn't that establish the terms are permissive?*

Short answer: he did far more than publish models — he redistributes the corpus
itself and ships weights inside an MIT-licensed Android library — and none of it
is a permission grant. Verdict below (§10.6).

### 10.1 Evidence inventory — what proshian actually publishes

**(a) The training repo re-hosts a preprocessed copy of the corpus.**
`proshian/neural-swipe-typing` README, verbatim:

> ### Option 2: Download the Preprocessed Dataset (Recommended)
> If you prefer to skip the lengthy preprocessing steps, you can directly
> download the preprocessed dataset:
> ```sh
> cd src
> python ./data_obtaining_and_preprocessing/download_dataset_preprocessed.py
> ```

<https://github.com/proshian/neural-swipe-typing#option-2-download-the-preprocessed-dataset-recommended>

That script is four lines and points at his own Google Drive, not at Yandex
(<https://raw.githubusercontent.com/proshian/neural-swipe-typing/main/src/data_obtaining_and_preprocessing/download_dataset_preprocessed.py>):

```python
import gdown

if __name__ == "__main__":
    DATA_PATH = "../data/data_preprocessed"

    url = "https://drive.google.com/drive/folders/1V2QxYfxkqHnMM3I-OJjYzlP5AgmMyiAN"

    gdown.download_folder(url, output=DATA_PATH, quiet=False, use_cookies=False)
```

That folder is live (HTTP 200, checked 2026-08-10) and its listing contains
`train_filtered.jsonl`, `valid.jsonl`, `test.jsonl`, `voc.txt`,
`gridname_to_grid.json`, `gridname_to_grid__fixed.json`,
`key_bounding_boxes.json`, `trajectory_features_statistics.json` — i.e. the
whole Yandex corpus, filtered and reformatted. No licence file, no terms, no
attribution notice accompanies it.

The DVC remote in the same repo is likewise a Google Drive folder
(`.dvc/config`: `url = gdrive://1OvqjaZKpSib_m6gCs1QvkfLILXKPiEs3`), and
`data/data_preprocessed/` holds DVC pointers for `train__default_only…jsonl`,
`train__extra_only…jsonl`, `valid.jsonl`, `test.jsonl`, `voc.txt`.

A **second** Drive re-host exists for the competition-reproduction path, cited in
`docs_and_assets/yandex_cup/submission_reproduciton_instrucitons.md`:

> В качестве альтернативы можно скачать результаты работы скрипта
> `./src/separate_grid.py` c [гугл диска](https://drive.google.com/drive/folders/1rRBUKUC0D6eZBJqT9qKs5fKQLl-gboej?usp=sharing)
> *(Alternatively you can download the output of `separate_grid.py` from Google Drive)*

*One point in his favour, checked explicitly:* the `default_only` / `extra_only`
split names refer to the two keyboard grids **inside the main archive**
(`task.md`: «grid_name – название раскладки (default или extra)»), not to the
second archive. Grepping his tree for the `accepted` archive's link
(`disk.yandex.ru/d/-qAoI9Ux1eP7XQ`) finds it **only** in the mirrored task
statement, never in a script. Like us (§5), he stayed on the elicited data.



**(b) Competition weights are published, also on Google Drive.**
`src/downloaders/download_weights.py`:

```python
url = "https://drive.google.com/drive/folders/1-iFPYCcRYy-tEu14Ry6xU6SMMf3eCjn6"
```

Live (HTTP 200); folder title `trained_models_for_final_submit`, containing the
`m1_bigger` checkpoints described in `docs_and_assets/report/solution_extra_info.md`.

**(c) A hosted inference service, whose source repo also commits corpus files.**
<https://proshian.pythonanywhere.com/> (HTTP 200, checked 2026-08-10) — a public
web keyboard running a model trained on the corpus. README: *"Try out a live
demo with a trained model from the competition through this web app"*. Its
source, `proshian/neuroswipe_inference_web` (GitHub API `license: None`), has
committed in-tree:

- `static/voc.txt` — **9,891,016 B, i.e. the Yandex vocabulary again, verbatim**
- `static/model_weights/m1_bigger_v2__2023_11_12__20_38_47__0.13129__greed_acc_0.86130__extra_l2_0_ls0_switch_2.pt`
  (4,931,613 B) — a competition checkpoint
- `static/nearest_key_lookup_state.pkl`, `static/gridname_to_grid.js` — corpus-derived layout artefacts

**(d) The sharpest item: an MIT-licensed Android library that ships the corpus
vocabulary verbatim and the trained weights.**
`proshian/neural-swipe-keyboard-android` (created 2025-01-11) carries a real
`LICENSE`:

> MIT License
> Copyright (c) 2025 Harry Proshian
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or **sell**
> copies of the Software…

Committed in that MIT repo's tree:

| path | size | what it is |
|---|---|---|
| `app/src/main/assets/logitProcessorResources/voc.txt` | 9,891,016 B | **byte-identical to the Yandex corpus `voc.txt`** |
| `app/src/main/assets/models/ru_default__xnnpack_my_nearest_feats.pte` | 4,705,968 B | ExecuTorch model trained on the corpus |
| `app/src/main/assets/models/ru_default__raw_my_nearest_feats.pte` | 4,652,312 B | same, unquantised |

(That is the **third** in-tree copy of the corpus vocabulary across his repos.)

The `voc.txt` identity is verified, not inferred: `sha256` of the committed file
and of `~/ctc-train/data/yandex_cup/voc.txt` are both
`b85623d0acf48183599d03750bfe1b9197c70611c47f2d1cc3157ced306e607b`, 503,598
lines, 9,891,016 bytes, `diff` clean. `trie-builder/.../Main.kt` builds the
shipped trie straight from it (`val vocab = File("app/src/main/assets/voc.txt").readLines()`),
which is why the README says *"the current trie (~170 MB) includes over 0.5
million Russian words"*.

And it is released as a **downloadable app**, not only as source
(<https://github.com/proshian/neural-swipe-keyboard-android/releases>):

| release | assets |
|---|---|
| `v0.1.0-alpha` (2025-04-02) | `neural-swipe-keyboard-0.1.0-alpha.apk`, `xnnpack_my_nearest_feats.pte`, `trie.ser` |
| `v0.1.0-alpha-2` (2025-04-21) | `neural-swipe-keyboard-0.1.0-alpha-2.apk`, `ru_default__xnnpack_my_nearest_feats.pte`, `trie.ser` |

So on the facts, proshian is **not** a conservative research-only actor. He does
two things this memo says not to do (§8.1, §8.5): he redistributes the corpus
(twice over, plus a DVC remote), and he ships corpus-derived assets — including
a byte-identical copy of a corpus file — inside a permissively-licensed,
commercially-usable software package with an installable APK.

**(e) What he does not do: state any terms, anywhere.**

- `proshian/neural-swipe-typing`: GitHub API `license: None`; no `LICENSE`,
  `NOTICE`, or `CITATION.cff` in the repo root (root listing is
  `.dvc .dvcignore .gitignore README.md configs data docs_and_assets requirements results src tokenizers`).
- README has no licence section, no data-usage statement, no attribution
  clause, no ethics note.
- `docs_and_assets/report/report.md` (435 lines, his full write-up): searched
  for `yandex|dataset|данн|licen|ethic|acknowledg|permis` — exactly one hit, and
  it is about model architecture ("I used this kind of custom transformer in
  Yandex Cup 23"). **No data provenance section, no permission statement, no
  ethics statement, no acknowledgements.**
- `docs_and_assets/yandex_cup/task/task.md` is the mirrored contest statement
  already quoted in §2.1 — it is the *source* of this memo's finding that no
  terms exist, not an independent grant.
- The MIT Android repo's README says nothing about what the model was trained on
  beyond *"The models are trained in a separate neural-swipe-typing repository"*
  — no dataset attribution, no asset-licence carve-out. The MIT text therefore
  reads, on its face, as covering the whole distribution including `voc.txt` and
  the `.pte` files. He is purporting to MIT-license material he did not create
  and holds no rights in.
- HuggingFace: he has an account (`huggingface.co/proshian`, HTTP 200) with
  three unrelated repos (`mts-ml-cup-2023`, `dgl_wheels`,
  `bitsandbytes-wheels-…`). **No swipe model, no swipe dataset card.** API search
  for `neuroswipe` across models and datasets: empty, both. This confirms §2.6
  and matters because HF is where a licence field would have been *forced* on
  him; Google Drive and GitHub release assets are exactly the channels that let
  you distribute without ever declaring terms.

### 10.2 The other competitors

- **kbrodt/yandex-cup-2023-neuroswipe** (1st place): GitHub API `license: None`.
  One release, `0.1` "Neuroswipe model", body *"Pretrained seq2seq model weights
  for neuroswipe"*, asset `models.zip` (163,622,521 B, 8 downloads); the README
  additionally points at a Yandex Disk mirror of the same weights
  (`yadi.sk/d/4oyVFBWxLXs-Pw`). For the data itself he says only "Download from
  the competition page" — so the winner **also publishes weights with no licence**,
  but he does **not** re-host the corpus and does **not** ship an app.
  His repo's single issue (#1 "Is there a way to contact you?", 18 comments,
  opened by a Grammarly engineer who later published Grammarly's own swipe-typing
  work) is a purely technical Q&A — sample counts, augmentation, optimizer, beam
  search, label smoothing. **Licensing is never mentioned by either party**, in
  the one substantial public conversation that exists about this dataset.
- Every other public NeuroSwipe repo found (`medbar/maa-neuroswipe2023`,
  `Valentin-Buchnev/neuroswipe-2023`, `Podpall/neuroswipe`, `kern/neuroswipe`,
  `Light-J/Neuroswipe`, `ASGusev/yandex_cup_2023_ml_neural_swipe`,
  `traptrip/yandex_cup_ml_2023`; `gubankov/yandex-cup-2023-swipe__dataset` is an
  empty repo) is **code only**: no weights, no data. `Podpall/neuroswipe` is MIT
  but its MIT covers scripts and a 179-byte character vocab — exactly the "MIT as
  to their own code" case already noted in §2.5.
- The Kaggle mirror's uploader is now identified: `sharthz23` = Daniel Potapov,
  Head of AI Lab at RSHB, Moscow — **a fellow participant, not Yandex**. Kaggle
  API returns `"licenseName":"Unknown"`, confirming §2.5 from the API rather than
  the rendered page.

So the population is: one actor (proshian) who redistributes data + weights +
an APK, one (kbrodt) who publishes weights only, and everyone else code-only.
proshian is an outlier, not a norm.

### 10.3 Two primary sources this re-review turned up that §§2–3 missed

Chasing the proshian trail back to its origin surfaced two Yandex-side documents
the original sweep did not reach. Both cut **against** permission, so §§2–3 were
right in conclusion while incomplete in coverage. Recording them here rather than
editing §§2–3, per the no-rewrite constraint.

**(i) «Условия использования сервиса "Яндекс.Контест"» — the platform ToU, and
the most on-point instrument found to date.**
<https://yandex.ru/legal/contest_termsofuse/>, published 01.10.2020, in force
throughout the contest. Accepted by conduct (cl. 1.3: «Начиная использовать
Сервис/его отдельные функции, Пользователь считается принявшим настоящие
Условия»). Verified directly, 2026-08-10. Clause 2.4, verbatim:

> «Права на дизайн Сервиса в целом и отдельные его элементы, права на
> программное обеспечение Сервиса, принадлежат Яндексу. **Права на все
> материалы, включенные в состав Сервиса, принадлежат их правообладателям.**
> … Любое их копирование, воспроизведение, переработка, **распространение**,
> доведение до всеобщего сведения … либо иное использование вне рамок
> возможностей, предоставляемых Сервисом, а также **любое их использование в
> коммерческих целях запрещается.**»

Clause 4.1, verbatim:

> «Сервис предоставляется Пользователю для **личного некоммерческого
> использования** в объеме, который соответствует уровню доступа Пользователя в
> Сервисе на момент использования Сервиса»

This is the *contest platform's own* terms — much closer to the distribution
event than the generic `yandex.ru/legal/rules` relied on in §3.1, and it says the
same thing in stronger words. Whether a Yandex Disk file linked *from* a problem
statement is «материал, включённый в состав Сервиса» is arguable, and an
adversary would argue it is. It does not change §3's conclusion; it removes one
of the softer places to stand.

**(ii) Yandex characterises the corpus as its own asset.** ML-track page,
2023 snapshot
(<http://web.archive.org/web/20231027143229/https://yandex.ru/cup/ml/>):

> «Все задачи построены на основе **реальных обезличенных данных Яндекса**.»
> *(All tasks are built on the basis of real anonymised Yandex data.)*

Yandex's own press release on the keyboard model
(<https://yandex.ru/company/news/02-06-23>) corroborates §5's provenance finding:

> «разработчики использовали семь миллионов примеров реальных свайпов,
> собранных с помощью **внутренней краудсорсинговой платформы** компании»

An assertion of ownership over «данные Яндекса» is the opposite of a release.

**(iii) Confirmation that no open re-release happened.** Re-checked
independently: `github.com/yandex` (155 repos — `yandex/mlcup` carries only
`asr/ cv/ nlp/ recsys`, no 2023 NeuroSwipe content), `github.com/yandex-research`
(57 repos, none swipe/keyboard/gesture), the HuggingFace `yandex` org
(`yambda`, `alchemist`, `mad-cars`, `wmt24-en-ru-rate`),
<https://research.yandex.com/datasets> (5 datasets, none swipe-related). And the
contrast §3.4 draws holds up on a second example: `yandex/geo-reviews-dataset-2023`
states «## Лицензия — Распространяется под лицензией MIT» with an
`opensource@yandex-team.ru` contact, and `yandex/yambda` on HF is tagged
`apache-2.0` with a press release. NeuroSwipe got none of that apparatus.

**Also noted:** the `additional_data.zip` link (§5, the Яндекс.Клавиатура
production set we deliberately avoid) is still live at 9.58 GB. Continue to
leave it alone.

### 10.4 What this establishes — and what it does not

**It does not establish permission.** Three independent reasons:

1. **Nemo dat.** proshian holds no rights in the Yandex database, so his MIT
   grant cannot convey any. A downstream user who relies on that MIT text for
   `voc.txt` or the `.pte` weights is relying on a licence from a non-owner —
   which is worth exactly nothing against ООО «Яндекс» under ст. 1334. If
   anything this *worsens* the picture: it shows the artefacts circulating with
   a licence label that is affirmatively wrong, so "there was an MIT licence on
   it" is not even an innocent-infringer story once you have read this memo.
2. **No counterparty ever spoke.** Across everything reviewed — 30 issues/PRs on
   `neural-swipe-typing` (all authored by proshian himself), kbrodt's single
   18-comment thread, the release bodies, the READMEs, the report, the mirrored
   task statement — **there is not one word from Yandex or anyone speaking for
   Yandex about post-competition use.** No blessing, no dataset re-release, no
   terms page. The §2 finding stands unchanged: the corpus was published with
   zero stated terms, and nothing has been added since.
3. **Conduct is not construction.** "Several hobbyists read the silence as
   permission" is a fact about *what people did*, not about *what the terms say*.
   The terms say nothing; the statutory default (ст. 1334) fills the gap, and it
   fills it against us. Widespread quiet non-enforcement of a default is the
   normal condition of low-salience infringement, not evidence of a licence.

**What it does establish, honestly stated:** enforcement risk in practice has so
far been zero for actors of this size. But calibrate "this size":

| | stars | forks | headline download counts |
|---|---|---|---|
| `proshian/neural-swipe-typing` | 18 | 3 | — |
| `proshian/neural-swipe-keyboard-android` | 6 | 0 | apk 29 / 5, `.pte` 5 / 0, `trie.ser` 2 / 2 |
| `kbrodt/yandex-cup-2023-neuroswipe` | 3 | 1 | `models.zip` 8 |

Five downloads of a model file is not a distribution event Yandex would ever
notice. **The absence of enforcement here proves close to nothing** — it is
consistent with "the terms permit it", with "the terms forbid it but nobody is
watching", and with "somebody at Yandex saw it and did not care", and the
evidence cannot distinguish those. Searched for the disconfirming case too: no
takedown touching this corpus, and no issue or discussion anywhere — in any of
these repos, on Kaggle, or in a GitHub-wide issue search — raising a licensing
concern about it. Nobody has ever asked the question in public.

**But "Yandex does not enforce" is the wrong generalisation, and this is the one
place the re-review makes the picture *worse*.** `github/dmca` contains roughly
two dozen Yandex notices, including `2022/03/2022-03-30-yandex.md` and
`2024/03/2024-03-11-yandex.md` (both verified present in the repo). Those two are
takedowns of *participants' own solutions to Yandex.Lyceum tasks*, asserting
exclusive rights over the educational materials. None touches NeuroSwipe or
Yandex Cup. So the record is not "Yandex ignores competition/educational
material" — it is "Yandex has a live DMCA practice in exactly this subject area
and has simply never been pointed at this corpus." That distinction matters,
because a GitHub DMCA notice is cheap, fast, and needs no Russian forum — it is a
far more plausible enforcement route than the ст. 1334 litigation §4 discusses,
and F-Droid removal is cheaper still.

Note also that a distribution channel with an actual reviewer changes this. None
of these artefacts has ever passed through one: proshian's APK is a GitHub
release, not F-Droid or Play. No third-party keyboard has adopted these Russian
weights either — a GitHub-wide issue search for `"neural-swipe-typing"` returns
15 threads, all technical and most of them in *our own* CleverKeys trackers;
none is a shipping keyboard integrating the Russian model. So the enforcement
record contains no observation of the case we actually care about.

### 10.5 Research framing vs product framing

The distinction the prior analysis turns on (ст. 1335.1 научные/образовательные
limb) survives contact with the evidence, but proshian sits on both sides of it:

- **Research side.** The work *is* academic, and now confirmed as such: Harry
  Proshian / Гарри Прошян, **ITMO University**, MSc thesis defended 11 June 2024,
  supervisor Sergey Nikolenko, titled «Распознавание слов, соответствующих
  траектории свайпа по клавиатуре смартфона» (61 pp.), classified on its own
  cover sheet as fundamental rather than applied research
  («Тема в области фундаментальных исследований: да / прикладных: нет»).
  ORCID <https://orcid.org/0009-0002-9435-6149>. The report closes: *"As a result
  of the research, a four-layer transformer with a new swipe point representation
  method was developed and trained… To test the method, a keyboard prototype was
  created in the form of a web application."* A thesis using a competition
  dataset is squarely inside «в личных, научных, образовательных целях».
  **That part of his conduct is fully consistent with §7(b) of this memo and
  licenses nothing beyond it.**

  The thesis is also the place a data-usage statement would live, and it does not
  contain one. Its entire treatment of provenance is a single sentence —

  > «В настоящей работе был использован датасет, представленный компанией Яндекс
  > на соревновании Yandex Cup 2023.»
  > *(In this work, a dataset presented by the Yandex company at the Yandex Cup
  > 2023 competition was used.)*

  — and a keyword sweep of the full text returns **zero** hits for `лиценз`,
  `соглашени`, `этик`, `персональн`, `согласи`, `авторск`, `правообладат`,
  `анонимиз`, and `GDPR`. There is also **no arXiv or DOI version** of the swipe
  work (the only Proshian paper on arXiv is an unrelated event-sequence short
  paper), his Habr profile has zero articles, and no blog or social post on the
  subject was found. He has written **nothing, anywhere, about whether he was
  permitted to use this data** — so there is no reasoning here to adopt or
  distinguish, only conduct.
- **Product side.** The Android library is not framed as research. Its README is
  a motivation-and-adoption pitch: *"Most keyboard apps from large tech companies
  log your swipe gestures to their servers… This project aims to help mobile
  developers build privacy-focused keyboards."* It is an MIT library, published
  as `neuralSwipeTyping`, with an APK, explicitly inviting third-party
  integration. That is the *same* posture as CleverKeys, minus the visibility.

So the proshian material does not contain a precedent for "research use is fine"
(nobody disputed that) plus a separate precedent for "and shipping is fine too".
It contains one person doing both under the same silence, having never analysed
the question in writing at all.

**And the actor who *did* analyse it went the other way.** FUTO — the one
comparable party with lawyers, a shipping keyboard and a real install base —
published *FUTO Swipe: Layout-Agnostic Neural Swipe Decoding* (arXiv:2606.25247),
released their own corpus `futo-org/swipe.futo.org` under **MIT** and
`futo-org/swipe-negatives` under **Apache-2.0**, and used the Yandex corpus for
**held-out evaluation only**, citing it as a bare URL with no licence — because
there is none to cite. Their own phrasing, *"the largest MIT-licensed swipe
corpus we are aware of"*, is itself a quiet acknowledgement that the Yandex one
is not licensed. Between a hobbyist who never thought about it and a keyboard
company that built its own corpus rather than train on this one, the second is
the precedent that resembles CleverKeys.

### 10.6 Verdict: **CONFIRMED**, with one qualification

**The eval-only recommendation (§8) stands.** Nothing in the proshian evidence
trail is a licence, a grant, or a statement by the rightsholder. The specific
thing that would have overturned it — an explicit Yandex permission or an
official re-release under stated terms — was searched for directly and **does
not exist**. The strongest artefact found (an MIT declaration over the corpus
`voc.txt`) is a licence from someone with no standing to give one, which is a
*liability signal, not a safe harbour*. On the two secondary questions the
re-review was asked to settle: the Yandex.Contest ToU (§10.3(i)) makes the terms
picture slightly *worse*, and the Yandex DMCA record (§10.4) makes the
enforcement picture slightly *worse*. Nothing found makes either better.

**Qualification — the precedent does shift the practical risk estimate, in one
narrow way.** §7(a) rated shipping HIGH. On the merits it remains HIGH: nothing
about the rights analysis moved. But we now have three years of observed
practice (Nov 2023 → Aug 2026) in which corpus-derived weights, the corpus
itself, and the verbatim vocabulary have sat in public, on GitHub, in an
installable APK, under a wrong licence label, without any visible reaction from
Yandex. The honest expected-value read is that *probability of enforcement is
low*. It is the *magnitude and shape* of the downside that keeps the
recommendation where it is, and that shape is worse for us than for him on every
axis that matters:

| | proshian | CleverKeys |
|---|---|---|
| visibility | 6 stars, 5 model downloads | F-Droid listing, real install base |
| channel | GitHub release, no reviewer | F-Droid, with an inclusion policy and a public issue tracker |
| licence of the shipped work | MIT (permissive; the wrongness is his to own) | **GPL-3.0** — freedom 0 promises *every* recipient commercial use, so we would be *warranting* a permission we know we do not have |
| author exposure | pseudonymous-ish hobby project | the user's own name on a published app |
| documentation | says nothing about the data | this memo exists; we would be shipping with documented knowledge |

That last row is the decisive asymmetry. proshian can plausibly claim he never
thought about it. After this document, we cannot. Shipping a GPL-3.0 artefact
whose only permission theory is non-commercial, while holding a written analysis
saying so, converts a low-probability rights problem into a
disclosure-and-good-faith problem with F-Droid and with downstream forkers — and
that one does not depend on Yandex doing anything at all.

**Operational deltas from this re-review (additions to §8, not replacements):**

1. **Do not treat proshian's Drive folders or his MIT repo as a laundered
   source.** Pulling `voc.txt`, the `.pte` weights, or `train_filtered.jsonl`
   from him is the same corpus with a false licence attached, plus a second-hand
   provenance chain we cannot audit. Our copy came straight from the Yandex Disk
   link with a checksum (§1) — keep it that way.
2. **Do not lift `voc.txt` as a wordlist.** §8.1 already said this; §10.1(d)
   shows exactly how easy the mistake is to make (it arrives disguised as a
   trie, an app asset, or an MIT-licensed file). Russian wordlists with clean
   licences exist (e.g. Hunspell/LibreOffice `ru_RU`, OpenCorpora); use one.
3. **Our own public fork is clean and should stay clean.**
   `github.com/tribixbite/neural-swipe-typing` is public and unlicensed
   (inherited from an unlicensed upstream — worth fixing separately). It tracks
   English swipelogs and English checkpoints plus inherited `.dvc` *pointer*
   files for the Yandex splits — pointers, not data. No Yandex-derived bytes are
   committed. Verified 2026-08-10; re-verify before any push that touches
   `data/` or `checkpoints/`.
4. **If the risk appetite ever changes, the lever is still §8.6** (a written
   permission from Yandex), not "proshian did it".
5. **Attribute the corpus wherever we report RU numbers.** ст. 1335.1's final
   sentence requires it when extracted material is made available to an
   unlimited circle of people, and it costs nothing: "Yandex Cup 2023 NeuroSwipe,
   evaluation only". None of the actors reviewed here does this.

**Sources added by this re-review** (all fetched 2026-08-10):
`proshian/neural-swipe-typing` README + `src/data_obtaining_and_preprocessing/download_dataset_preprocessed.py`
+ `src/downloaders/download_weights.py` + `.dvc/config` + `docs_and_assets/report/report.md`
+ `docs_and_assets/yandex_cup/submission_reproduciton_instrucitons.md` ·
<https://github.com/proshian/neural-swipe-keyboard-android> (LICENSE, README,
`app/src/main/assets/**`, releases) ·
<https://github.com/proshian/neuroswipe_inference_web> (`static/**`) ·
<https://drive.google.com/drive/folders/1V2QxYfxkqHnMM3I-OJjYzlP5AgmMyiAN> ·
<https://drive.google.com/drive/folders/1-iFPYCcRYy-tEu14Ry6xU6SMMf3eCjn6> ·
<https://drive.google.com/drive/folders/1rRBUKUC0D6eZBJqT9qKs5fKQLl-gboej> ·
<https://proshian.pythonanywhere.com/> ·
<https://github.com/kbrodt/yandex-cup-2023-neuroswipe/issues/1> ·
<https://yandex.ru/legal/contest_termsofuse/> ·
<http://web.archive.org/web/20231027143229/https://yandex.ru/cup/ml/> ·
<https://yandex.ru/company/news/02-06-23> ·
<https://github.com/yandex/geo-reviews-dataset-2023> ·
`github/dmca` → `2022/03/2022-03-30-yandex.md`, `2024/03/2024-03-11-yandex.md` ·
ITMO MSc thesis, Прошян Г., 11.06.2024 ·
<https://orcid.org/0009-0002-9435-6149> ·
<https://huggingface.co/futo-org/swipe.futo.org> (MIT) ·
<https://huggingface.co/futo-org/swipe-negatives> (Apache-2.0)
