using Microsoft.Extensions.Configuration;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace SkSampleScrumPlanning
{
    public class Settings
    {
        private readonly IConfigurationRoot configRoot;

        private AzureOpenAISettings azureOpenAI;
        private OpenAISettings openAI;

        public AzureOpenAISettings AzureOpenAI => this.azureOpenAI ??= this.GetSettings<Settings.AzureOpenAISettings>();
        public OpenAISettings OpenAI => this.openAI ??= this.GetSettings<Settings.OpenAISettings>();

        public class OpenAISettings
        {
            public string ChatModel { get; set; } = string.Empty;
            public string ApiKey { get; set; } = string.Empty;
        }

        public class AzureOpenAISettings
        {
            public string ChatModelDeployment { get; set; } = string.Empty;
            public string Endpoint { get; set; } = string.Empty;
            public string ApiKey { get; set; } = string.Empty;
        }

        public TSettings GetSettings<TSettings>() =>
            this.configRoot.GetRequiredSection(typeof(TSettings).Name).Get<TSettings>()!;

        public Settings()
        {
            this.configRoot =
                new ConfigurationBuilder()
                    .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true)
                    .AddEnvironmentVariables()
                    .AddUserSecrets(Assembly.GetExecutingAssembly(), optional: true)
                    .Build();
        }
    }

}
