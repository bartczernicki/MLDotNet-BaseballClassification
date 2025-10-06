using Microsoft.Data.SqlClient;
using Microsoft.Extensions;
using nietras.SeparatedValues;
using System.Globalization;
using System.ClientModel;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Configuration.UserSecrets;

namespace MLBDataTablesExport
{
    internal class Program
    {
        private static readonly string ExportDir = @"C:\Users\BartAIServer\Downloads\Exports\"; // hard-coded
        private static readonly string Schema = "dbo";
        private static readonly string[] Tables =
        {
        "MLBBaseballBatters",
        "MLBBaseballBattersFullTraining",
        "MLBBaseballBattersHistorical",
        "MLBBaseballBattersHistoricalPositionPlayers",
        "MLBBaseballBattersPositionPlayers",
        "MLBBaseballBattersSplitTest",
        "MLBBaseballBattersSplitTraining"
        };

        static async Task Main(string[] args)
        {
            // Read connection string from user secrets
            var config = new ConfigurationBuilder()
                .AddUserSecrets<Program>()
                .Build();

            var connStr = config["SQLServerConnectionString"];
            if (string.IsNullOrWhiteSpace(connStr))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine(" No connection string found in user secrets (SQLServerConnectionString).");
                Console.ResetColor();
                return;
            }

            Directory.CreateDirectory(ExportDir);
            Console.WriteLine($" Export directory: {ExportDir}");
            Console.WriteLine($" Connecting to: {connStr}");

            await using var conn = new SqlConnection(connStr);
            await conn.OpenAsync();

            foreach (var table in Tables)
            {
                var qualified = $"[{Schema}].[{table}]";
                var path = Path.Combine(ExportDir, $"{table}.csv");
                Console.WriteLine($"\nExporting {qualified}");
                Console.WriteLine($"\n Path: {path}");

                try
                {
                    await ExportWithSepAsync(conn, qualified, path);
                    Console.WriteLine($" {table} done");
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine($" {table} failed: {ex.Message}");
                    Console.ResetColor();
                }
            }

            Console.WriteLine("\nAll exports complete");
        }

        private static async Task ExportWithSepAsync(SqlConnection conn, string qualifiedTable, string csvPath)
        {
            string sql = $"SELECT * FROM {qualifiedTable} WITH (NOLOCK);";
            await using var cmd = new SqlCommand(sql, conn);
            await using var reader = await cmd.ExecuteReaderAsync(System.Data.CommandBehavior.SequentialAccess);

            var sep = Sep.New(',');
            await using var writer = sep.Writer(o => o with { WriteHeader = true, Escape = true })
                                       .ToFile(csvPath);

            int fieldCount = reader.FieldCount;
            string[] colNames = new string[fieldCount];
            for (int i = 0; i < fieldCount; i++)
                colNames[i] = reader.GetName(i);

            var inv = CultureInfo.InvariantCulture;
            long rowCount = 0;

            while (await reader.ReadAsync())
            {
                // Cache IsDBNull checks before creating the row to avoid ref struct across await
                bool[] nullFlags = new bool[fieldCount];
                object?[] values = new object?[fieldCount];

                for (int i = 0; i < fieldCount; i++)
                {
                    nullFlags[i] = await reader.IsDBNullAsync(i);
                    if (!nullFlags[i])
                        values[i] = reader.GetValue(i);
                }

                // Now work with the row (ref struct) without any await calls
                using var row = writer.NewRow();
                for (int i = 0; i < fieldCount; i++)
                {
                    if (nullFlags[i])
                        continue;

                    var col = row[colNames[i]];
                    var val = values[i];
                    switch (val)
                    {
                        case DateTime dt:
                            col.Set(dt.ToString("yyyy-MM-dd HH:mm:ss.fffffff", inv));
                            break;
                        case DateTimeOffset dto:
                            col.Set(dto.ToString("yyyy-MM-dd HH:mm:ss.fffffff zzz", inv));
                            break;
                        case byte[] bytes:
                            col.Set(Convert.ToBase64String(bytes));
                            break;
                        case IFormattable f:
                            col.Set(f.ToString(null, inv));
                            break;
                        default:
                            col.Set(val?.ToString() ?? string.Empty);
                            break;
                    }
                }

                rowCount++;
                //if (rowCount % 10_000 == 0)
                //    Console.WriteLine($" {rowCount:N0} rows...");
            }

            Console.WriteLine($" {rowCount:N0} rows written");
        }
    }
}
