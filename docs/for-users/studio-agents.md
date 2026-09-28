# Studio Agents

Open a project in the Graph editor and click the Agent button in the graph toolbar. Studio opens an Agent window bound to that project; you can keep it beside the graph while the model works. Select a configured model provider and create a session. The model can inspect the installed node catalog, edit the graph, analyze and update a Python Script node's `code` field, deploy a project, and inspect runtime monitors and recent logs. Graph patches, code writes, deployments, and Unity installation require approval in the session. Review the proposed patch, code change, or installation preview shown in session artifacts before approving.

The deterministic provider remains available as a fixed graph-building example. Select a configured model provider for open-ended graph, code, and game tasks. Provider credentials stay in the Studio server environment.

The agent can list and read workflow skills. Studio includes `graph_python` and `unity_modding`. Add a game-specific skill at `<Studio data directory>/agent-skills/<skill-id>/SKILL.md`; the skill ID may contain lowercase letters, digits, underscores, and hyphens. A local skill with the same ID overrides a bundled skill. Skills provide workflow context; game-directory changes still go through Studio's typed preview and approval tools.

Unity targets can be detected, previewed, installed through the managed setup tool, and checked for decoded UDP skeleton frames. Unreal detection is available, but Studio does not yet provide a verified UE4SS installer. The agent cannot install an Unreal patch through Studio until that tool exists.

Code changes are committed to the project graph, not to source files in the repository. A code write requires the graph revision and source hash returned by the last code read; a conflicting edit must be read and reconciled again. Unsaved text in another browser editor window is not visible to the agent, so save or discard that draft before approving a code change.
