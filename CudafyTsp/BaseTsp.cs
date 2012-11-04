using System;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace CudafyTsp {
	public abstract class BaseTsp {
		// Count permutations
		static BaseTsp() { for (int i = 2; i <= _cities; i++) { _permutations *= i; } }

		public static int  NumCities			{ get { return _cities; } }
		public static int ThreadsPerBlock	{ get { return _threadsPerBlock; } }
		public static int BlocksPerGrid		{ get { return _blocksPerGrid; } }
		public static long Permutations		{ get { return _permutations; } }

      protected const int _cities				= 13;
      protected const int _threadsPerBlock	= 32 /*threads/warp*/ * 4 /*warps/block*/;
      protected const int _blocksPerGrid		=  6 /*blocks / gpu*/ * 2 /* gpus*/ * 8*4*2 /*parallelization*/; 
      protected static readonly long _permutations	= 1;

		public long LoadTime	{ get; protected set; }
		public string Name	{ get { return GetType().Name.Replace("GpuTsp",string.Empty); } }
	}

	[Cudafy]
	public struct LatLongStruct {
		public float Latitude;
		public float Longitude;

		public float Distance(LatLongStruct other) {
         return (float)GMath.Sqrt(Square(Latitude  - other.Latitude) 
											+ Square(Longitude - other.Longitude));
		}
		public float Square(float value) { return value * value; }
	}

	public abstract class AbstractTsp : BaseTsp {
      public static readonly float[] _latitudes		= new float[_cities];
      public static readonly float[] _longitudes	= new float[_cities];

		internal abstract Answer GetAnswer();

		/// <summary>Build Cities.</summary>
		public static void BuildCityData(Action<int, float, float> action) {
			var cities = Cities.List;
         var j = 0;
         foreach (var fields in cities
				.Split('\r')
				.Select(city => city.Replace("\r", "").Split(',') )
				.Where( (fields,i) => (fields.Length==3 && i<_cities) )
			) {
				action(j++, float.Parse(fields[1]), float.Parse(fields[2]));
         }
		}
	}
	public abstract class AbstractTspCPU : AbstractTsp {
		#region Cudafy
		[Cudafy]
      public static float FindPathDistance(long permutations, long permutation, int cities, 
			float[] latitudes, float[] longitudes, int[,] paths, int pathIndex) {
         PathFromRoutePermutation(permutations, permutation, cities, paths, pathIndex);

         float distance = 0;
         var city = paths[pathIndex, 0];
         var previousLatitude = latitudes[city];
         var previousLongitude = longitudes[city];

         for (var i = 1; i < cities; i++)
         {
               city = paths[pathIndex, i];
               var latitude = latitudes[city];
               var longitude = longitudes[city];
               distance += (float)Math.Sqrt(Square(latitude - previousLatitude) 
														+ Square(longitude - previousLongitude));
               previousLatitude = latitude;
               previousLongitude = longitude;
         }

         return distance;
      }

      [Cudafy]
      public static float Square(float value) { return value * value; }

      [Cudafy]
      public static float PathFromRoutePermutation(long permutations, long permutation, int cities, 
			int[,] paths, int pathIndex) {
         for (var i = 0; i < cities; i++) { paths[pathIndex, i] = i; }

			// Credit: SpaceRat. 
			// http://www.daniweb.com/software-development/cpp/code/274075/all-permutations-non-recursive
         for (long remaining = cities, divisor = permutations; remaining > 0; remaining--) {
            divisor /= remaining;
            var index = (permutation / divisor) % remaining;

            var swap = paths[pathIndex, index];
            paths[pathIndex, index] = paths[pathIndex, remaining - 1];
            paths[pathIndex, remaining - 1] = swap;
         }

         return 0;
		}
		#endregion
	}
}
