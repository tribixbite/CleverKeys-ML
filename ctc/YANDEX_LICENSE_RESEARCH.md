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
