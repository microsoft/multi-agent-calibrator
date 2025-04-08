# The Scrum Planning multi-agent sample business logic
- The Scrum Planning app is a simple application that allows users to create and manage Scrum planning sessions. It provides a user-friendly interface for creating, editing, and deleting planning sessions, as well as adding and removing participants. The app also includes features for tracking the progress of each session and generating reports on the overall performance of the team.
- The Agents include,
	- ScrumChampAgent: This agent is responsible for managing the Scrum planning sessions. It creates and manages the sessions, assigns tasks to participants, and tracks the progress of each session. 
	- ProgramManagerAgent: This agent is responsible for managing the overall program. It creates and manages the project, assigns tasks to participants, and tracks the progress of each session.
		- Scrum Board Plugin:
			- Team Availability Function:
	- EngineerAgent: This agent is responsible for managing the engineering team. It creates and manages the engineering team, assigns tasks to participants, and tracks the progress of each session.
	- TesterAgent: This agent is responsible for managing the testing team. It creates and manages the testing team, assigns tasks to participants, and tracks the progress of each session.

# Run locally
- Rename 'appsettings.sample.json' to appsettings.json, and ensure 'Copy Always' for this file in Visual Studio.
  - Replace the settings inside the file to your own setting.
- Grant permission to your account to the AOAI resource
  - Assign role "Cognitive Services OpenAI Contributor" to your account on <AOAI-NAME> in Azure
- Auth: In Visual Studio, 'Tools' => 'Options' => 'Azure Service Authentication' => 'Account Selection' choose your account you want to use during auth.