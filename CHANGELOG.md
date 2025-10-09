# [v0.11.0](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.11.0)
*Released on 2025-10-09T08:51:56Z*

- added: OCR text indexing
    - you can now chat with your note attachments too
- added: doc/query conditioning (embeddings v3)
    - this is expected to improve semantic search
    - you will be prompted to rebuild your Jarvis database
- added: keep response text selected for accept/reject/regenerate
- added: research: pubmed database paper search
- added: setting: autocomplete prompt
- added: setting: notes embeddings timeout
- improved: upgraded models list
    - gpt-5
    - claude 4.1
    - gemini 2.5
- improved: html note processing
- improved: embeddings error handling: report note ID / title, retry / skip note
- improved: openai error message handling
- improved: research: paper ranking with new settings
- improved: research: prompts and otuput
- improved: decrease default temperature setting
- fixed: claude-opus support
- fixed: claude max_tokens setting
- chore: move most logs to debug log

**Full Changelog**: https://github.com/alondmnt/joplin-plugin-jarvis/compare/v0.10.6...v0.11.0

---

# [v0.10.6-alpha](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.10.6)
*Released on 2025-09-19T13:04:11Z*

- improved: embedding error handling
- added: setting: notes_embed_timeout
- package update: google/generative-ai
- chore: move most logs to debug log

---

# [v0.10.5](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.10.5)
*Released on 2025-08-27T13:05:02Z*

- added: 'None' model to disable generation features (#56)
- improved: error handling for JSON responses in OpenAI model queries (#54)

---

# [v0.10.4](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.10.4)
*Released on 2025-07-02T02:33:07Z*

- improved: RTE support for chat / autocomplete (#50)
    - some issues remain with output formatting

---

# [v0.10.3](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.10.3)
*Released on 2025-06-21T01:06:20Z*

- fixed: regression: the default chat model setting displayed OpenAI-custom endpoint but was gpt-4o-mini behind the scenes (#47 #48)

---

# [v0.10.2](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.10.2)
*Released on 2025-05-24T10:10:18Z*

- new: setting: `Annotate: Keep existing tags`

---

# [v0.10.1](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.10.1)
*Released on 2025-05-24T09:50:34Z*

- fixed: Gemini prompts always begin with user role (system message)
- improved: updated OpenAI models: gpt-4.1-nano, gpt-4.1-mini, gpt-4.1, o4-mini, o3
- improved: updated Claude models: claude-sonnet-4-0, claude-opus-4-0
- improved: updated Gemini models: gemini-2.5-flash, gemini-2.5-pro
- improved: error handling in Gemini
- improved: Ollama embeddings
    - support API key (#45)
    - support native, openai and legacy APIs
- improved: scroll cursor to line
- new: setting: `Notes: Scroll delay (ms)`

---

# [v0.10.0](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.10.0)
*Released on 2025-03-14T14:10:31Z*

- models
    - new: Claude 3.5 and 3.7, and support for custom models (OpenAI-compatible endpoints) (FR #34)
    - new: Gemini 2.0 models (flash, flash-lite, pro)
    - new: GPT-4.5, o1, and support for other custom o-models (OpenAI-compatible endpoints)
    - deprecated legacy models: GPT-3.5 (still available as a custom model), Gemini 1
    - improved: removed constraint on `max_tokens`, `memory_tokens`, `notes_context_tokens`
        - previously, `max_tokens`, `memory_tokens` and `notes_context_tokens` were tied together due to how models used to calculate their tokens
        - many models today separate context, or input prompt, from the length of the generated output / response
        - so it is now up to the user to ensure that the settings match the model's limits
    - improved: model descriptions, separating input tokens from output tokens
- new: cancel button for database updates
- new: setting: `Notes: Abort on error` (FR #39)
- improved: model error handling
- improved: reset default API key to empty strings (FR #33)
- improved: USE model follows rate limit settings
- improved: UI style

**Full Changelog**: https://github.com/alondmnt/joplin-plugin-jarvis/compare/v0.9.1...v0.10.0

---

# [v0.9.1](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.9.1)
*Released on 2024-11-02T10:40:26Z*

- improved: ignore html notes
    - Jarvis currently processes properly into blocks / chunks only Markdown notes

**Full Changelog**: https://github.com/alondmnt/joplin-plugin-jarvis/compare/v0.9.0...v0.9.1

---

# [v0.9.0](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.9.0)
*Released on 2024-08-14T03:31:24Z*

- models
    - new: default model `gpt-4o-mini`
    - deprecated: `gpt-4-turbo`
    - new: support for [Ollama embeddings API](https://github.com/alondmnt/joplin-plugin-jarvis/blob/master/GUIDE.md#offline-embedding-model-with-olalma)
- chat with notes / related notes
    - new: settings to customise what extra information goes into each block / chunk (default: all selected)
        - title
        - full headings path
        - last heading
        - note tags
    - improve: chat with notes prompt and model compatibility
- annotations
    - new: setting `Annotate: Preferred language`

---

# [v0.8.5](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.8.5)
*Released on 2024-06-04T01:58:52Z*

- new: separate settings sections for chat, related notes, annotations and research
- fix: set default values for API keys
    - this is a workaround that ensures that keys are saved securely to your keychain (where available)
- changed default settings
    - context tokens: increased to 2048
    - annotation: tags method changed to existing tags

---

# [v0.8.4](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.8.4)
*Released on 2024-05-15T12:06:44Z*

- fix: work with Joplin <v3

---

# [v0.8.3](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.8.3)
*Released on 2024-05-15T00:08:30Z*

- **note that the min Joplin app version for this release is v3.0**
- improve: exclude notes in trash from db
- improve: OpenAI model updates
    - added `gpt-4o` (latest model)
    - deprecated legacy `gpt-3.5-turbo-16k`
        - `gpt-3.5-turbo` points to a newer version of this model
    - deprecated legacy `gpt-4`
    - all legacy models are still accessible via the `openai-custom` model setting
    - improved model descriptions with tokens / price category

---

# [v0.8.0](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.8.0)
*Released on 2024-05-04T03:48:55Z*

## new features
- revamped `Edit with Jarvis` command (@jakubjezek001) **(screenshot below)**
- `Auto-complete with Jarvis` command to autocomplete any text at the current cursor position
- scroll to line of a found note chunk from the panel
- chat context preview dialog **(screenshot below)**
- token counter command
- display note similarity score in panel

## new models
- OpenAI
    - replace `text-davinci` (deprecated) models with `gpt-3.5-turbo-instruct` (@jakubjezek001)
    - 3rd generation embedding / notes models `text-embedding-3-small` and `text-embedding-3-large`
    - chat model `gpt-4-turbo`: an efficient, strong model with a context window of 128K tokens
- Google AI
    - deprecated PaLM
    - chat models `gemini-1-pro` and `gemini-1.5-pro` (a strong model with a context window of 1M tokens!)
    - embedding / notes models `embedding-001` and `text-embedding-004`

## new settings
- `Notes: Context tokens`: the number of context tokens to extract from notes in "Chat with your notes" (previously used `Chat: Memory tokens`)
- `Notes: Context history`: the number of user prompts to base notes context on for "Chat with your notes"
- `Notes: Custom prompt`: the prompt (or additional instructions) to use for generating "Chat with your notes" responses
- `Notes: Parallel jobs`: the number of parallel jobs to use for calculating text embeddings

## chat improvements
- chat display format **(screenshot below)**
- chat with notes default prompt
- chat parsing

## general improvements
- CodeMirror 6 / beta editor support
- load USE from cache instead of re-downloading every time
- faster model test on startup / model switch
- various fixes

## ux
- new standard dialog style

Screenshot 1: New edit dialog
<img width="50%" alt="image" src="https://github.com/alondmnt/joplin-plugin-jarvis/assets/17462125/9cebde5d-4f1d-478b-b4ae-7250696835fd">

Screenshot 2: New chat context preview
<img width="70%" alt="image" src="https://github.com/alondmnt/joplin-plugin-jarvis/assets/17462125/0afd22d8-8292-4755-9b60-0a22abfc32e8">

Screenshot 3: New chat display format
<img width="50%" alt="image" src="https://github.com/alondmnt/joplin-plugin-jarvis/assets/17462125/3d2a5dc9-2692-4bbf-adcf-e127cc3c76a4">

---

# [v0.8.0-alpha.2](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.8.0-alpha.2)
*Released on 2024-04-20T14:31:41Z*

pre-release with a number of features planned for v0.8.0. bold: added in this release.

- new features
    - revamped note edit command interface (@jakubjezek001)
    - scroll to line of a found note chunk from the panel
    - chat context preview dialog
    - **token counter command**
    - display note similarity score in panel
- new settings
    - `Notes: Context tokens`: the number of context tokens to extract from notes in "Chat with your notes" (previously used `Chat: Memory tokens`)
    - **`Notes: Context history`: the number of user prompts to base notes context on for "Chat with your notes"**
    - **`Notes: Custom prompt`: the prompt (or additional instructions) to use for generating "Chat with your notes" responses**
    - **`Notes: Parallel jobs`: the number of parallel jobs to use for calculating text embeddings**
- chat improvements
    - **chat display format**
    - chat with notes default prompt
    - chat parsing
- general improvements
    - CodeMirror 6 / beta editor support
    - load USE from cache instead of re-downloading every time
    - faster model test
- new models
    - replace `text-davinci` models with `gpt-3.5-turbo-instruct` (@jakubjezek001)
- ux
    - new standard dialog style

---

# [v0.8.0-alpha.1](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.8.0-alpha.1)
*Released on 2024-04-18T01:11:25Z*

pre-release with a number of features planned for v0.8.0.

- new features
    - revamped note edit command interface (@jakubjezek001)
    - scroll to line of a found note chunk from the panel
    - chat context preview dialog
    - display note similarity score in panel
- new settings
    - `Notes: Context tokens`: the number of context tokens to extract from notes in "Chat with your notes" (previously used `Chat: Memory tokens`)
- chat improvements
    - default chat context based on the last user prompt
    - chat with notes prompt
    - chat parsing
- general improvements
    - CodeMirror 6 / beta editor support
    - load USE from cache instead of re-downloading every time
    - faster model test
- new models
    - replace `text-davinci` models with `gpt-3.5-turbo-instruct` (@jakubjezek001)
- ux
    - new standard dialog style

---

# [v0.7.0](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.7.0)
*Released on 2023-08-31T19:09:10Z*

Jarvis can now work **completely offline!** (Continue reading)
This release adds two new model interfaces.

## Google PaLM

- If you have [access](https://makersuite.google.com/app/apikey) to it (it's free), you can use it for chat and for related notes.

## Custom OpenAI-like APIs

- This allows Jarvis to use custom endpoints and models that have an OpenAI-compatible interface.
- Example: [tested] [OpenRouter](https://openrouter.ai) (for ebc000) [setup guide](https://github.com/alondmnt/joplin-plugin-jarvis/blob/master/GUIDE.md#openrouter)
- Example: [not tested] [Azure OpenAI](https://oai.azure.com/) (previously [requested](https://github.com/alondmnt/joplin-plugin-jarvis/issues/9)) 
- Example: [tested] Locally served [GPT4All](https://gpt4all.io) (for laurent, and everyone else who showed interest) [setup guide](https://github.com/alondmnt/joplin-plugin-jarvis/blob/master/GUIDE.md#offline-chat-model-with-gpt4all)
    - This is an open source, offline model (you may in fact choose from several available models), that you can install and run on a laptop. It can be used for chat, and potentially also for related notes (embeddings didn't work for me, probably due to a gpt4all issue, but related notes already support the USE offline model).
    - This solution for an offline model is not ideal, as it may be technically challenging for a user to run their own server, but at the moment this workaround looks like the only viable solution, and doesn't involve a lot of steps.
- Example: [not tested] [LocalAI](https://github.com/go-skynet/LocalAI)
    - This is another self-hosted server that supports many models, in case you run into issues with GPT4All.

**Full Changelog**: https://github.com/alondmnt/joplin-plugin-jarvis/compare/v0.6.0...v0.7.0

---

# [v0.6.0](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.6.0)
*Released on 2023-08-09T20:57:50Z*

- Annotations
    - This release introduces the toolbar button / command `Annotate note with Jarvis`. It can automatically annotate a note based on its content in 4 ways: By setting the title of the note; by adding a summary section; by adding links to related notes; and by adding tags. (`gpt-4` is recommended for tags.) Each of these 4 features can be turned on or off in the settings in order to customize the behavior of the command. In addition, each sub-command can be run separately. For more information see [this guide](https://github.com/alondmnt/joplin-plugin-jarvis/blob/master/GUIDE.md#annotate-note-with-jarvis).
- System message
    - You may edit it in the settings to inform Jarvis who he is, what is his purpose, and provide more information about yourself and your interests, in order to customize Jarvis' responses.

**Full Changelog**: https://github.com/alondmnt/joplin-plugin-jarvis/compare/v0.5.3...v0.6.0

---

# [v0.5.3](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.5.3)
*Released on 2023-07-20T19:10:09Z*

- new: custom OpenAI model IDs (closes #12)
    - select `Chat: Model` "(online) OpenAI: custom"
    - in the Advanced Settings section, set `Chat: OpenAI custom model ID`, for example: `gpt-4-0314`

---

# [v0.5.2](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.5.2)
*Released on 2023-07-19T19:04:59Z*

- new: search box in the related notes panel
    - use free text to semantically search for related notes
    - in the example below, the notes are sorted by their relevance to the query in the box
        - within each note, its sections are sorted by their relevance
    - you may hide it in the settings ("Notes: Show search box")

<img width="200" alt="image" src="https://github.com/alondmnt/joplin-plugin-jarvis/assets/17462125/8c9f1bcf-425b-4f55-838a-578a902628cb">

- new: global commands for chat with notes
    - any command that appears in a "jarvis" code block will set its default behavior for the current chat / note
    - you may override this default by using the command again within a specific prompt in the chat
    - for example:

                ```jarvis
                Context: This is the default context for each prompt in the chat.
                Search: This is the default search query.
                Not context: This is the default text that will be excluded from semantic search, and appended to every prompt.
                ```

**Full Changelog**: https://github.com/alondmnt/joplin-plugin-jarvis/compare/v0.5.1...v0.5.2

---

# [v0.5.1](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.5.1)
*Released on 2023-07-09T16:49:38Z*

- changed default Hugging Face embedding model to `paraphrase-multilingual-mpnet-base-v2`

---

# [v0.5.0](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.5.0)
*Released on 2023-07-09T15:16:37Z*

This release includes many improvements under the hood, such as better chat processing, error handling, note chunking, and token counting (for example, no need to set max tokens in the settings in most cases). Most importantly, this version introduces new models for document embedding (related notes) and text generation (ask, research, chat). Adding other models in the future will be easier.

Personally, for the time being, I'm probably going to stay with the offline USE for related notes and the online `gpt-3.5-turbo-16k` for chat. The latter model that was introduced recently by OpenAI provides a large context window (high `Max tokens`) that is great for text summarization (including literature reviews / research), and for chatting with your notes (by increasing `Memory tokens` you can get a lot more context into the conversation).

- **New chat / ask / research models**
    - Anything on Hugging Face (deafult: `LaMini-Flan-T5-783M`)
        - The great news are that HF has a *free* inference API (no need to set an API key)
        - But do note, that the free version only supports the smaller (less sophisticated) models, yet they're still worth a try. The default model should work fast and reasonably well, all considered, but it's not the state of the art. If you're willing to pay for compute, you can get access to all hosted models
    - OpenAI
        - Extended-context `gpt-3.5-turbo-16k`
        - Everyone should also have access now to the long-awaited `gpt-4`
        - Deprecated a few old models of GPT-3

- **New semantic search / related notes models (embeddings)**
    - OpenAI (`text-embedding-ada-002`)
        - As far as I know, this model can [process code blocks](https://github.com/openai/openai-cookbook/blob/main/examples/Code_search.ipynb). In order to enable code processing, check the setting "Notes: Include code blocks in DB"
        - This model is multi-lingual. I could not find official documentation for this, but it is likely to work OK with many European languages. It doesn't seem to support Asian languages that well
    - Anything on Hugging Face (default: `paraphrase-xlm-r-multilingual-v1`)
        - I had good experience with HF embeddings
        - The default model is multi-lingual (trained on 50+ languages), and works well for a diverse set of languages
    - Some comments
        - These online models may be faster than the default (offline) model if you're using an old machine
        - You can try out a few models and switch between them. Jarvis will index your notes in a separate database for each and load the relevant results
        - You may need to adjust the setting `Minimal note similarity` when switching to a new model, in order to get the most relevant results displayed in the Related Notes panel
        - I couldn't get an additional offline model to work properly (and specifically [transformers.js](https://huggingface.co/docs/transformers.js/index)), perhaps next time (PRs are welcome)

- **Chat with notes features**
    - Added a new [user guide](https://github.com/alondmnt/joplin-plugin-jarvis/blob/master/GUIDE.md)
    - The note database version has changed, and you will be asked to upgrade the DB
    - Only commands that appear in the last user prompt apply
    - New commands: `Context:` and `Not Context:`
    - Improved `Search:` command so that selected blocks must contain the search keywords
    - Here is a recap of all available chat commands:

|        Command |                                                                                                         Description | Content included<br>in Jarvis prompt | Content included<br>in context search |
|----------------|---------------------------------------------------------------------------------------------------------------------|----------------------------------|-----------------------------------|
|      ` Notes:` |                                   The following list of note IDs (or internal<br>links) will be the source for chat context |                               No |                               Yes |
|      `Search:` |   The following Joplin search query will<br>be used to search for related notes<br>(in addition to semantic search), and<br>search keywords must appear in the<br>selected context |                               No |                               Yes |
|     `Context:` |   The following text will be the one<br>used to semantically search for related<br>notes instead of the entire note |                               No |                               Yes |
| `Not Context:` | The following text will be excluded<br>from semantic search (e.g., it can be used<br>to define Jarvis' role), but the rest of the<br> conversation will still be used |                              Yes |                                No |

**Full Changelog**: https://github.com/alondmnt/joplin-plugin-jarvis/compare/v0.4.4...v0.5.0

---

# [v0.4.4](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.4.4)
*Released on 2023-05-22T19:14:55Z*

This release adds a number of experimental features that help you shape the context that Jarvis gets from your notes. Most of these are disabled by default until we gather more feedback, so look them up in the settings.

- **preview command** (`Tools-->Jarvis-->Preview chat notes context`) (following @davadev 's feature request)
    - see the exact context that will be sent to the model previewed in the related notes panel
    - all of the new features below will be reflected in the preview, so you can experiment with these settings and see how they affect the resulting chat context
- **prompt commands**
    - `Notes:` this command, when it appears in the chat prompt (in a new line), specifies the note IDs for the context (following @davadev 's feature request)
    - `Search:` this command, when it appears in the chat prompt (in a new line), uses Joplin search to retrieve relevant notes
    - you may combine both commands in your prompts (see example)
    - hard constraint: this mechanism replaces the usual search for related notes based on chat content and gives you (almost) complete control over the context that is selected. that is, only notes that come up in search, or listed in notes will be considered

```markdown
User: Hi Jarvis, please summarize information on finger spins in the following notes.
Notes: [abcd](:/9b3e075fea954195b79d4238132dbe3b), 456f02ffc4984ef7bc9c117f1589f3dd
Search: yoyo competition tag:2018
```

- **note similarity and chat context can take links into account** (following @davadev 's feature request)
    - soft constraint: the content of the chat/note is still taken into account, but any links that appear in the note will be used to refine the search for related notes. that is, there is no guarantee that the linked notes will be selected to be included in the chat context
    - set the weight given to links in the settings (start with values in the range 20-50)
- **extension of the blocks (note chunks) sent to GPT**
    - attaching the previous X blocks in the note to each selected block
    - attaching the next X blocks in the note
    - attaching the X most similar notes to the block in the DB
    - check the advanced settings to set these, and increase the "Memory tokens" setting to squeeze more blocks into the prompt
- **minimal block length**
    - set to 100 chars by default (can be adjusted in the settings)
    - shorter blocks will be excluded
- **welcome message before initializing the note DB** (following @SteveShank 's issue)
    - with explanations and an option to postpone the process

**Full Changelog**: https://github.com/alondmnt/joplin-plugin-jarvis/compare/v0.4.3...v0.4.4

---

# [v0.4.3](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.4.3)
*Released on 2023-05-13T17:47:38Z*

- new: setting to include / exclude code blocks
- improved: max tokens warnings

**Full Changelog**: https://github.com/alondmnt/joplin-plugin-jarvis/compare/v0.4.2...v0.4.3

---

# [v0.4.2](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.4.2)
*Released on 2023-05-12T06:29:17Z*

**Full Changelog**: https://github.com/alondmnt/joplin-plugin-jarvis/compare/v0.4.1...v0.4.2

---

# [v0.4.1](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.4.1)
*Released on 2023-05-10T22:28:05Z*

**Full Changelog**: https://github.com/alondmnt/joplin-plugin-jarvis/compare/v0.4.0...v0.4.1

---

# [v0.4.0](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.4.0)
*Released on 2023-05-10T22:27:26Z*

- **Related notes**
    - Find notes based on semantic similarity to the currently open note, or to selected text. This is done locally, without sending the content of your notes to a remote server. Notes are displayed in a dedicated panel. To run semantic search based on selected text, click on the `Find related notes` (toolbar button or context menu option).
    - Coincidentally, this turned out to be semantically similar to a [longstanding plugin](https://github.com/marcgreen/semantic-joplin/tree/master/similar-notes), but adds some technical improvements.
    - I will add support for multilingual online models in the next release, but I wanted this to be an offline feature first, as some users may feel uncomfortable sending their entire Joplin database to a third party. I believe that in most cases this will also run faster.
    - The current offline model (Google's Universal Sentence Encoder) performs well, but its main drawback is that it only supports English (sorry, I'm not a native either).

- **Chat with your notes**
    - To add additional context to your conversation based on your notes, select the command `Chat with your notes` (from the Tools/Jarvis menu). Relevant short excerpts from your notes will be sent to OpenAI in addition to the usual conversation prompt / context. To exclude certain notes from this feature, add the tag `#exclude.from.jarvis` to the notes you wish to exclude. The regular chat is still available, and will not send out any of your notes. You may switch between regular chat and note-based chat on the same note.

- Improved token length estimation.

- Typo fix by @Wladefant.

**Full Changelog**: https://github.com/alondmnt/joplin-plugin-jarvis/compare/v0.3.2...v0.4.0

---

# [v0.3.2](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.3.2)
*Released on 2023-03-30T19:21:41Z*

- new: added `gpt-4` / `gpt-4-32k` as optional models (closes #5)
    - note that currently this will work only if you have [early access](https://openai.com/waitlist/gpt-4-api) to these APIs, until the models become publicly available. also note the considerably [higher pricing](https://openai.com/pricing) for these models (compared to `gpt-3.5-turbo`).
- new: handling different max_tokens limits per model
- new: chat indication that a response is being generated (closes #4) 
- fix: better token length estimation (closes #3) 
- changed: AGPLv3 license

**Full Changelog**: https://github.com/alondmnt/joplin-plugin-jarvis/compare/v0.3.1...v0.3.2

---

# [v0.3.1](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.3.1)
*Released on 2023-03-17T18:53:02Z*

- improved: wikipedia summary
- changed: default temperature to 10
- fix: handle empty search results
- fix: prompt typos

**Full Changelog**: https://github.com/alondmnt/joplin-plugin-jarvis/compare/v0.3.0...v0.3.1

---

# [v0.3.0](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.3.0)
*Released on 2023-03-12T23:03:14Z*

Jarvis is now connected to the web! (if you choose to)

- **added: new model set as default**
    - the `gpt-3.5-turbo` model is the one behind ChatGPT. its responses are a lot cheaper and are much faster (at least they used to be in the first few days). check the settings to see that you have it selected. this is great timing, because for the next new feature Jarvis needs to make many queries. personally, the switch was not as smooth as I anticipated, and it took me a couple of days to adjust my prompts in order to get good responses from the model. in the end, though, I'm mostly happy with the new results.

![jarvis-research](https://user-images.githubusercontent.com/17462125/224579200-c207266c-f3b6-453c-b31d-befefb0d03df.gif)

- **added: new command "Research with Jarvis"**
    - "Research with Jarvis" generates automatic academic literature reviews. just write what you're interested in as free text, and optionally adjust the search parameters (high `max_tokens` is recommended). wait 2-3 minutes for all the output to appear in the note (depending on internet traffic). Jarvis will update the content as it finds new information on the web (using Semantic Scholar, Crossref, Elsevier, Springer & Wikipedia databases). in the end you will get a report with the following sections: title, prompt, research. questions, queries, references, review and follow-up questions. this is not Bing AI or the cool [Elicit](https://elicit.org) project, but even a small DIY tool can do quite a lot with the help of a large language model. I'll write more about how it's done in the future.
    - sources of information: Jarvis currently supports 2 search engines (and Wikipedia), and uses various paper/abstract repositories. as a general rule, you're likely to get better results when operating from a university campus or IP address, because institutions usually have access to more papers. the 2 search engines have complementary features, and I recommend trying both.
        - **Semantic Scholar:** (default) search is usually faster, more flexible (likely to find something related to any prompt), and it requires no API key. however, it has a tendency to prefer obscure, uncited papers.
        - **Scopus:** search is slower, and stricter, but tends to find higher impact papers. it requires registering for a [free API key](https://dev.elsevier.com/apikey/manage)
    - Jarvis is a Joplin assistant first and foremost, but this feature was also ported to a [VSCode extension](https://marketplace.visualstudio.com/items?itemName=alondmnt.jarvis-notes).

- **contributions:** thanks to @ryanfreckleton for fixing a prompt typo (#2).

**Full Changelog**: https://github.com/alondmnt/joplin-plugin-jarvis/compare/v0.2.0...v0.3.0

---

# [v0.2.0](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.2.0)
*Released on 2023-02-14T17:10:20Z*

## chat with Jarvis

- new command: Chat with Jarvis (Cmd+Shift+C). this is the homemade equivalent of ChatGPT within a Joplin note. each time you run the command Jarvis will append its response to the note at the current cursor position (given the previous content that both of you created). or it will try to extend its own response if you didn't add anything. therefore, this essentially serves as a general-purpose autocomplete command. repeat the command a few times to replace the response with a new one.

![YuzuCheeseCake](https://user-images.githubusercontent.com/17462125/218807777-d0138356-3d01-4f7e-9afa-9c29694d0d1a.gif)

## prompt templates

- new: predefined and customizable prompt templates (check the settings). quickly select a combination of an instruction, a format for the response, a character that Jarvis will play, and perhaps add a nudge towards thinking more analytically than usual. this utilizes some of the techniques from this [recommended tutorial](https://learnprompting.org), and draws inspiration from @sw-yx's very cool [reverse prompting project](https://github.com/sw-yx/ai-notes/blob/main/Resources/Notion%20AI%20Prompts.md). the new prompts are a work in progress. help me improve them, and add new useful templates to the database.
- new: select whether to show or hide the input prompt in the response.

![prompt_templates](https://user-images.githubusercontent.com/17462125/218807928-e8494bf3-1476-47f0-adcf-576a92bf7fa5.jpg)

## stability

- improved: lost queries display error messages with info from OpenAI.
- new: auto-retry sending the query to OpenAI (adjusted if needed) when pressing the OK button in the error dialog.

## look and feel

- improved: dark theme support.
- new: dedicated Jarvis sub-menu under Tools.

**Full Changelog**: https://github.com/alondmnt/joplin-plugin-jarvis/compare/v0.1.3...v0.2.0

---

# [v0.1.3](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.1.3)
*Released on 2022-12-07T20:08:21Z*

- improved: dialog style

---

# [v0.1.2](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.1.2)
*Released on 2022-11-29T07:18:54Z*

- improve: added `text-davinci-003` as the default model

---

# [v0.1.1](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.1.1)
*Released on 2022-11-22T19:23:39Z*

- fixed: increased the default creativity level of the model (temperature=9).

---

# [v0.1.0](https://github.com/alondmnt/joplin-plugin-jarvis/releases/tag/v0.1.0)
*Released on 2022-11-20T20:17:08Z*

First release.

---
