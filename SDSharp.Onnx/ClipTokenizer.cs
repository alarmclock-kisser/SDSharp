using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace SDSharp.Onnx
{
    internal sealed class ClipTokenizer
    {
        private const string StartOfTextToken = "<|startoftext|>";
        private const string EndOfTextToken = "<|endoftext|>";
        private static readonly Regex TokenRegex = new(
            "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
            RegexOptions.Compiled | RegexOptions.CultureInvariant | RegexOptions.IgnoreCase);

        private readonly Dictionary<string, int> Encoder;
        private readonly Dictionary<(string Left, string Right), int> BpeRanks;
        private readonly Dictionary<string, string> Cache = new(StringComparer.Ordinal);
        private readonly Dictionary<byte, string> ByteEncoder;

        public int StartOfTextTokenId { get; }
        public int EndOfTextTokenId { get; }
        public int MaxLength { get; } = 77;



        public ClipTokenizer(string vocabJson, string mergesText, int maxLength = 77)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(vocabJson);
            ArgumentException.ThrowIfNullOrWhiteSpace(mergesText);

            this.Encoder = JsonSerializer.Deserialize<Dictionary<string, int>>(vocabJson)
                ?? throw new InvalidOperationException("Tokenizer vocabulary could not be parsed.");
            this.BpeRanks = ParseMerges(mergesText);
            this.ByteEncoder = CreateByteEncoder();
            this.MaxLength = maxLength;

            if (!this.Encoder.TryGetValue(StartOfTextToken, out int sot))
            {
                throw new InvalidOperationException($"Tokenizer vocabulary does not contain '{StartOfTextToken}'.");
            }

            if (!this.Encoder.TryGetValue(EndOfTextToken, out int eot))
            {
                throw new InvalidOperationException($"Tokenizer vocabulary does not contain '{EndOfTextToken}'.");
            }

            this.StartOfTextTokenId = sot;
            this.EndOfTextTokenId = eot;
            this.Cache[StartOfTextToken] = StartOfTextToken;
            this.Cache[EndOfTextToken] = EndOfTextToken;
        }



        public long[] EncodeText(string? text)
        {
            string normalized = WhitespaceClean(BasicClean(text ?? string.Empty)).ToLowerInvariant();
            var tokens = new List<int>(this.MaxLength) { this.StartOfTextTokenId };

            foreach (Match match in TokenRegex.Matches(normalized))
            {
                if (!match.Success || string.IsNullOrEmpty(match.Value))
                {
                    continue;
                }

                string encodedToken = this.EncodeBytesToUnicode(match.Value);
                string bpe = this.ApplyBpe(encodedToken);

                foreach (string part in bpe.Split(' ', StringSplitOptions.RemoveEmptyEntries))
                {
                    if (this.Encoder.TryGetValue(part, out int tokenId))
                    {
                        tokens.Add(tokenId);
                    }
                    else
                    {
                        tokens.Add(this.EndOfTextTokenId);
                    }

                    if (tokens.Count >= this.MaxLength - 1)
                    {
                        break;
                    }
                }

                if (tokens.Count >= this.MaxLength - 1)
                {
                    break;
                }
            }

            tokens.Add(this.EndOfTextTokenId);

            while (tokens.Count < this.MaxLength)
            {
                tokens.Add(this.EndOfTextTokenId);
            }

            return tokens.Select(static id => (long) id).ToArray();
        }



        private string EncodeBytesToUnicode(string token)
        {
            byte[] utf8Bytes = Encoding.UTF8.GetBytes(token);
            var builder = new StringBuilder(utf8Bytes.Length);

            foreach (byte value in utf8Bytes)
            {
                builder.Append(this.ByteEncoder[value]);
            }

            return builder.ToString();
        }



        private string ApplyBpe(string token)
        {
            if (this.Cache.TryGetValue(token, out string? cached))
            {
                return cached;
            }

            if (string.IsNullOrEmpty(token))
            {
                return string.Empty;
            }

            var word = token[..^1].Select(ch => ch.ToString()).ToList();
            word.Add(token[^1] + "</w>");

            var pairs = GetPairs(word);
            while (pairs.Count > 0)
            {
                (string Left, string Right)? candidate = null;
                int bestRank = int.MaxValue;

                foreach ((string left, string right) in pairs)
                {
                    if (this.BpeRanks.TryGetValue((left, right), out int rank) && rank < bestRank)
                    {
                        bestRank = rank;
                        candidate = (left, right);
                    }
                }

                if (candidate == null)
                {
                    break;
                }

                var newWord = new List<string>(word.Count);
                int index = 0;
                while (index < word.Count)
                {
                    int nextIndex = word.FindIndex(index, item => item == candidate.Value.Left);
                    if (nextIndex < 0)
                    {
                        newWord.AddRange(word[index..]);
                        break;
                    }

                    newWord.AddRange(word[index..nextIndex]);
                    index = nextIndex;

                    if (index < word.Count - 1 && word[index] == candidate.Value.Left && word[index + 1] == candidate.Value.Right)
                    {
                        newWord.Add(word[index] + word[index + 1]);
                        index += 2;
                    }
                    else
                    {
                        newWord.Add(word[index]);
                        index++;
                    }
                }

                word = newWord;
                if (word.Count == 1)
                {
                    break;
                }

                pairs = GetPairs(word);
            }

            string result = string.Join(' ', word);
            this.Cache[token] = result;
            return result;
        }



        private static Dictionary<(string Left, string Right), int> ParseMerges(string mergesText)
        {
            var ranks = new Dictionary<(string Left, string Right), int>();
            int rank = 0;

            foreach (string line in mergesText.Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries))
            {
                string trimmed = line.Trim();
                if (trimmed.Length == 0 || trimmed.StartsWith("#", StringComparison.Ordinal))
                {
                    continue;
                }

                string[] parts = trimmed.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length != 2)
                {
                    continue;
                }

                ranks[(parts[0], parts[1])] = rank++;
            }

            return ranks;
        }



        private static HashSet<(string Left, string Right)> GetPairs(IReadOnlyList<string> word)
        {
            var pairs = new HashSet<(string Left, string Right)>();
            if (word.Count < 2)
            {
                return pairs;
            }

            string previous = word[0];
            for (int i = 1; i < word.Count; i++)
            {
                string current = word[i];
                pairs.Add((previous, current));
                previous = current;
            }

            return pairs;
        }



        private static string BasicClean(string text)
        {
            return WebUtility.HtmlDecode(WebUtility.HtmlDecode(text)).Replace('\u00A0', ' ');
        }



        private static string WhitespaceClean(string text)
        {
            return Regex.Replace(text, "\\s+", " ").Trim();
        }



        private static Dictionary<byte, string> CreateByteEncoder()
        {
            var bytes = new List<int>();
            bytes.AddRange(Enumerable.Range('!', '~' - '!' + 1));
            bytes.AddRange(Enumerable.Range('¡', '¬' - '¡' + 1));
            bytes.AddRange(Enumerable.Range('®', 'ÿ' - '®' + 1));

            var chars = new List<int>(bytes);
            int extra = 0;
            for (int b = 0; b < 256; b++)
            {
                if (bytes.Contains(b))
                {
                    continue;
                }

                bytes.Add(b);
                chars.Add(256 + extra);
                extra++;
            }

            var mapping = new Dictionary<byte, string>(256);
            for (int i = 0; i < bytes.Count; i++)
            {
                mapping[(byte) bytes[i]] = char.ConvertFromUtf32(chars[i]);
            }

            return mapping;
        }
    }
}
