using System.ComponentModel;
using System.Net;
using System.Text.Json.Serialization;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;

using Azure.Identity;
using SkSampleScrumPlanning;

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

        // Create a kernel with Azure OpenAI chat completion
        var chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();

        // Add a plugin (the LightsPlugin class is defined below)
        kernel.Plugins.AddFromType<ScrumBoardPlugin>("ScrumBoard");

        // Enable planning
        OpenAIPromptExecutionSettings openAIPromptExecutionSettings = new()
        {
            FunctionChoiceBehavior = FunctionChoiceBehavior.Auto()
        };

        // Create a history store the conversation
        var history = new ChatHistory();
        history.AddUserMessage("Help me find the team capacity for current sprint");

        // Get the response from the AI
        var result = await chatCompletionService.GetChatMessageContentAsync(
           history,
           executionSettings: openAIPromptExecutionSettings,
           kernel: kernel);

        // Print the results
        Console.WriteLine("Assistant > " + result);

        // Add the message from the agent to the chat history
        history.AddAssistantMessage(result.Content);
    }
}
