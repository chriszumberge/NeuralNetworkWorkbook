using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;
using System.Xml;

namespace NeuralNetwork.DecisionTree
{
    public static class Extensions
    {
        public static string Serialize<T>(this T obj)
        {
            using (var memoryStream = new MemoryStream())
            {
                var serializer = new DataContractSerializer(obj.GetType());
                serializer.WriteObject(memoryStream, obj);
                return Encoding.UTF8.GetString(memoryStream.ToArray());
            }
        }

        public static T Deserialize<T>(this string xml)
        {
            using (var memoryStream = new MemoryStream(Encoding.UTF8.GetBytes(xml)))
            {
                var reader = XmlDictionaryReader.CreateTextReader(memoryStream, Encoding.UTF8, new XmlDictionaryReaderQuotas(), null);
                var serializer = new DataContractSerializer(typeof(T));
                return (T)serializer.ReadObject(reader);
            }
        }

        /// <summary>
        /// Returns all distinct elements of the given source, where "distinctness"
        /// is determined via a projection and the default equality comparer for the projected type.
        /// </summary>
        /// <remarks>
        /// This operator uses deferred execution and streams the results, although
        /// a set of already-seen keys is retained. If a key is seen multiple times,
        /// only the first element with that key is returned.
        /// </remarks>
        /// <typeparam name="TSource">Type of the source sequence</typeparam>
        /// <typeparam name="TKey">Type of the projected element</typeparam>
        /// <param name="source">Source sequence</param>
        /// <param name="keySelector">Projection for determining "distinctness"</param>
        /// <returns>A sequence consisting of distinct elements from the source sequence,
        /// comparing them by the specified key projection.</returns>

        public static IEnumerable<TSource> DistinctBy<TSource, TKey>(this IEnumerable<TSource> source,
            Func<TSource, TKey> keySelector)
        {
            return source.DistinctBy(keySelector, null);
        }

        /// <summary>
        /// Returns all distinct elements of the given source, where "distinctness"
        /// is determined via a projection and the specified comparer for the projected type.
        /// </summary>
        /// <remarks>
        /// This operator uses deferred execution and streams the results, although
        /// a set of already-seen keys is retained. If a key is seen multiple times,
        /// only the first element with that key is returned.
        /// </remarks>
        /// <typeparam name="TSource">Type of the source sequence</typeparam>
        /// <typeparam name="TKey">Type of the projected element</typeparam>
        /// <param name="source">Source sequence</param>
        /// <param name="keySelector">Projection for determining "distinctness"</param>
        /// <param name="comparer">The equality comparer to use to determine whether or not keys are equal.
        /// If null, the default equality comparer for <c>TSource</c> is used.</param>
        /// <returns>A sequence consisting of distinct elements from the source sequence,
        /// comparing them by the specified key projection.</returns>

        public static IEnumerable<TSource> DistinctBy<TSource, TKey>(this IEnumerable<TSource> source,
            Func<TSource, TKey> keySelector, IEqualityComparer<TKey> comparer)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            if (keySelector == null) throw new ArgumentNullException(nameof(keySelector));
            return DistinctByImpl(source, keySelector, comparer);
        }

        private static IEnumerable<TSource> DistinctByImpl<TSource, TKey>(IEnumerable<TSource> source,
            Func<TSource, TKey> keySelector, IEqualityComparer<TKey> comparer)
        {
            var knownKeys = new HashSet<TKey>(comparer);
            foreach (var element in source)
            {
                if (knownKeys.Add(keySelector(element)))
                {
                    yield return element;
                }
            }
        }
    }
}
