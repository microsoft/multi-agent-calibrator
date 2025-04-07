using System.ComponentModel;
using System.Net;
using System.Text.Json.Serialization;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Azure.Identity;
using SkSampleScrumPlanning;
using System.Diagnostics;
using Microsoft.SemanticKernel.Agents.Chat;
using Microsoft.SemanticKernel.Agents;
using Microsoft.SemanticKernel.Connectors.AzureOpenAI;
using System.Text.Json;


public class ScrumBoardPlugin
{
    [KernelFunction("get_team_capacity")]
    [Description("Gets team capacity for current sprint")]
    public async Task<int> GetLightsAsync()
    {
        return 30;
    }
}


public static class Program
{
    private sealed class ClipboardAccess
    {
        [KernelFunction]
        [Description("Copies the provided content to the clipboard.")]
        public static void SetClipboard(string content)
        {
            if (string.IsNullOrWhiteSpace(content))
            {
                return;
            }

            using Process clipProcess = Process.Start(
                new ProcessStartInfo
                {
                    FileName = "clip",
                    RedirectStandardInput = true,
                    UseShellExecute = false,
                });

            clipProcess.StandardInput.Write(content);
            clipProcess.StandardInput.Close();
        }
    }

    public static async Task Main()
    {
        // Load configuration from environment variables or user secrets.
        Settings settings = new();

        var builder = Kernel.CreateBuilder();
        Kernel kernel = builder.AddAzureOpenAIChatCompletion(
            settings.AzureOpenAI.ChatModelDeployment,
            settings.AzureOpenAI.Endpoint,
            new AzureCliCredential())
            .Build(); ;

        Kernel toolKernel = kernel.Clone();
        toolKernel.Plugins.AddFromType<ClipboardAccess>();

        const string ScrumChampName = "ScrumChamp";
        const string ProgramManagerName = "ProgramManager";

        ChatCompletionAgent agentScrumChamp =
            new()
            {
                Name = ScrumChampName,
                Instructions =
                    """
                    You are ScrumChampAgent, the dedicated manager of Scrum planning sessions. Your responsibilities include:

                    - **Session Management:** Create, edit, and delete Scrum planning sessions.
                    - **Participant Coordination:** Add or remove participants as needed and assign specific tasks to team members.
                    - **Progress Tracking:** Continuously monitor and report on the progress of each planning session.
                    - **Utilize Available Tools:** Leverage the Scrum Board Plugin, including the Team Availability Function, to support efficient session management.

                    Your goal is to ensure that each Scrum planning session runs smoothly and aligns with Scrum best practices.
                    
                    """,
                Kernel = toolKernel,
                Arguments = new KernelArguments(new AzureOpenAIPromptExecutionSettings() { FunctionChoiceBehavior = FunctionChoiceBehavior.Auto() })
            };

        ChatCompletionAgent agentProgramManager =
            new()
            {
                Name = ProgramManagerName,
                Instructions =
                    """
                    You are ProgramManagerAgent, responsible for overseeing the overall project management within the Scrum Planning app. Your duties include:

                    - **Project Oversight:** Create and manage the overall project structure, ensuring that all elements of Scrum planning are integrated.
                    - **Task Assignment:** Assign tasks to participants across all planning sessions.
                    - **Performance Monitoring:** Track the progress of each session and ensure they contribute to the project's broader objectives.
                    - **Tool Integration:** Utilize the Scrum Board Plugin to gain insights into team availability and session performance.

                    Your objective is to ensure that the project remains well-coordinated, with clear accountability and measurable progress in every session.
                    
                    """,
                Kernel = kernel,
            };

        agentProgramManager.Kernel.Plugins.AddFromType<ScrumBoardPlugin>("ScrumBoard");

        KernelFunction selectionFunction =
           
             AgentGroupChat.CreatePromptFunctionForStrategy(
                $$$"""
                Examine the provided RESPONSE and choose the next participant.
                State only the name of the chosen participant without explanation.
                Never choose the participant named in the RESPONSE.

                Choose only from these participants:
                - {{{ScrumChampName}}}
                - {{{ProgramManagerName}}}

                Always follow these rules when choosing the next participant:
                - If RESPONSE is user input, it is {{{ScrumChampName}}}'s turn.
                - If RESPONSE is by {{{ScrumChampName}}}, it is {{{ProgramManagerName}}}'s turn.
                - If RESPONSE is by {{{ProgramManagerName}}}, it is {{{ScrumChampName}}}'s turn.

                RESPONSE:
                {{$lastmessage}}
                """,
                safeParameterNames: "lastmessage");

        const string TerminationToken = "yes";

        KernelFunction terminationFunction =
            AgentGroupChat.CreatePromptFunctionForStrategy(
                $$$"""
                Examine the RESPONSE and determine whether the content has been deemed satisfactory.
                If content is satisfactory, respond with a single word without explanation: {{{TerminationToken}}}.
                If specific suggestions are being provided, it is not satisfactory.
                If no correction is suggested, it is satisfactory.

                RESPONSE:
                {{$lastmessage}}
                """,
                safeParameterNames: "lastmessage");

        ChatHistoryTruncationReducer historyReducer = new(1);

        AgentGroupChat chat =
            new(agentScrumChamp, agentProgramManager)
            {
                ExecutionSettings = new AgentGroupChatSettings
                {
                    SelectionStrategy =
                        new KernelFunctionSelectionStrategy(selectionFunction, kernel)
                        {
                            // Always start with the editor agent.
                            InitialAgent = agentScrumChamp,
                            // Save tokens by only including the final response
                            HistoryReducer = historyReducer,
                            // The prompt variable name for the history argument.
                            HistoryVariableName = "lastmessage",
                            // Returns the entire result value as a string.
                            ResultParser = (result) => result.GetValue<string>() ?? agentScrumChamp.Name
                        },
                    TerminationStrategy =
                        new KernelFunctionTerminationStrategy(terminationFunction, kernel)
                        {
                            // Only evaluate for editor's response
                            Agents = [agentScrumChamp],
                            // Save tokens by only including the final response
                            HistoryReducer = historyReducer,
                            // The prompt variable name for the history argument.
                            HistoryVariableName = "lastmessage",
                            // Limit total number of turns
                            MaximumIterations = 12,
                            // Customer result parser to determine if the response is "yes"
                            ResultParser = (result) => result.GetValue<string>()?.Contains(TerminationToken, StringComparison.OrdinalIgnoreCase) ?? false
                        }
                }
            };

        Console.WriteLine("Ready!");

        bool isComplete = false;
        do
        {
            Console.WriteLine();
            Console.Write("> ");
            string input = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(input))
            {
                continue;
            }
            input = input.Trim();
            if (input.Equals("EXIT", StringComparison.OrdinalIgnoreCase))
            {
                isComplete = true;
                break;
            }

            if (input.Equals("RESET", StringComparison.OrdinalIgnoreCase))
            {
                await chat.ResetAsync();
                Console.WriteLine("[Conversation has been reset]");
                continue;
            }

            if (input.StartsWith("@", StringComparison.Ordinal) && input.Length > 1)
            {
                string filePath = input.Substring(1);
                try
                {
                    if (!File.Exists(filePath))
                    {
                        Console.WriteLine($"Unable to access file: {filePath}");
                        continue;
                    }
                    input = File.ReadAllText(filePath);
                }
                catch (Exception)
                {
                    Console.WriteLine($"Unable to access file: {filePath}");
                    continue;
                }
            }

            chat.AddChatMessage(new ChatMessageContent(AuthorRole.User, input));

            chat.IsComplete = false;

            try
            {
                await foreach (ChatMessageContent response in chat.InvokeAsync())
                {
                    Console.WriteLine();
                    Console.WriteLine($"{response.AuthorName.ToUpperInvariant()}:{Environment.NewLine}{response.Content}");
                }
            }
            catch (HttpOperationException exception)
            {
                Console.WriteLine(exception.Message);
                if (exception.InnerException != null)
                {
                    Console.WriteLine(exception.InnerException.Message);
                    if (exception.InnerException.Data.Count > 0)
                    {
                        Console.WriteLine(JsonSerializer.Serialize(exception.InnerException.Data, new JsonSerializerOptions() { WriteIndented = true }));
                    }
                }
            }
        } while (!isComplete);

    }
}
