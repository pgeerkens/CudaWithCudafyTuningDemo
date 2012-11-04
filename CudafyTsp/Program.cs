using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace CudafyTsp
{
   internal partial class Program {
      private static void Main() {
			#region Platform
#if x64
			var target = "x64";
#elif x86
			var target = "x86";
#else
			var target = "Any";
#endif
#if DEBUG
			target += " Debug";
#else
			target += " Release";
#endif
			#endregion
			target += string.Format(" ( {0} threads_per * {1} blocks = {2} threads)",
				BaseTsp.ThreadsPerBlock, BaseTsp.BlocksPerGrid, BaseTsp.ThreadsPerBlock*BaseTsp.BlocksPerGrid );
			var path = Path.GetFullPath(Directory.GetCurrentDirectory() + Path.DirectorySeparatorChar);
			File.Delete(path + typeof(Tsp)+".cdfy");
			File.Delete(path + typeof(GpuTsp1_SeparateClass).Name+".cdfy");
			File.Delete(path + typeof(GpuTsp2_StructArray).Name+".cdfy");

			File.Delete(path + typeof(GpuTsp3_Architecture_x64_2_1).Name+".cdfy");
			File.Delete(path + typeof(GpuTsp3_PathArrayStrided).Name+".cdfy");
			File.Delete(path + typeof(GpuTsp3_DivisorsCachedGlobal).Name+".cdfy");
			File.Delete(path + typeof(GpuTsp3_MultiplyInstead).Name+".cdfy");

			File.Delete(path + typeof(GpuTsp4_Long).Name+".cdfy");
			File.Delete(path + typeof(GpuTsp4_PathArrayStrided).Name+".cdfy");
			File.Delete(path + typeof(GpuTsp4_DivisorsCachedGlobal).Name+".cdfy");
			File.Delete(path + typeof(GpuTsp4_MultiplyInstead).Name+".cdfy");
			CudafyHost.GetDevice().FreeAll();

			Console.WriteLine(string.Format(target + @"
Cities {0,3};   Permutations: {1,10:D}:
---------------------------------------", BaseTsp.NumCities, BaseTsp.Permutations));
			if (BaseTsp.NumCities < 13) {
				Console.WriteLine("With disk cache empty and only a single class ...");
				Console.WriteLine("           Total     Load      Run            ");
				Console.WriteLine("              ms       ms       ms            distance");
				if (BaseTsp.NumCities < 12) {
					Console.WriteLine("CpuTsp  " + Tsp.CpuTsp());
					Console.WriteLine("MpuTsp  " + Tsp.MpuTsp());
					var mpuTsp = new MpuTsp_Better();
					Console.WriteLine("MpuTspA " + mpuTsp.GetAnswer() + " - " + mpuTsp.Name);
				}
				Console.WriteLine("GpuTsp0 " + Tsp.GpuTsp() + " - cold");
				File.Delete(path + typeof(Tsp).Name+".cdfy");
				Console.WriteLine("GpuTsp0 " + Tsp.GpuTsp() + " - warm");
				Console.WriteLine("");
				Console.WriteLine("... now with separate classes");
			}

			Console.WriteLine("           Total     Load      Run            ");
			Console.WriteLine("              ms       ms       ms            distance");
			Tsp.FindRoute(Cities.List);

			if (BaseTsp.NumCities < 13) {
				Console.WriteLine("... and now with disk cache populated.");
				Console.WriteLine("           Total     Load      Run            ");
				Console.WriteLine("              ms       ms       ms            distance");
				Tsp.FindRoute(Cities.List);
			}

         Console.WriteLine("Done ... Press Enter to shutdown.");
			try { Console.Read(); } catch (InvalidOperationException) { ; }
			CudafyHost.GetDevice().FreeAll();
			CudafyHost.GetDevice().HostFreeAll();
      }
   }

   internal class Answer  {
      public float Distance;
      public long Permutation;
		public long msLoadTime;
      public long msRunTime;

      public override string ToString() {
			int[] path = RoutePermutation();
			var s = "";
			foreach (int c in path) s+= string.Format("{0},",c);

         return string.Format("{0,8} = {1,6} + {2,6}; solution: {3}: ({5})", 
				msLoadTime+msRunTime, msLoadTime, msRunTime, Distance, 
				FindPathDistance(path,AbstractTsp._latitudes, AbstractTsp._longitudes), s);
      }
		protected virtual int[] RoutePermutation() {
			int cities = Tsp.NumCities;
			int[] path = new int[cities];
			for (var i = 0; i < cities; i++) { path[i] = i; }

			// Credit: SpaceRat. 
			// http://www.daniweb.com/software-development/cpp/code/274075/all-permutations-non-recursive
			var divisor = Tsp.Permutations;
			for (int city = cities; city > 0; city--) {
				divisor /= city;
				var index = (Permutation / divisor) % city;

				var swap			= path[index];
				path[index]		= path[city - 1];
				path[city - 1]	= swap;
			}
			return path;
		}
      public static float FindPathDistance(int[] paths,
			float[] latitudes, float[] longitudes) {

			Func<float,float> Square = (v) => (v*v);
         float distance = 0;
         var city = paths[0];
         var previousLatitude = latitudes[city];
         var previousLongitude = longitudes[city];

         for (var i = 1; i < paths.Length; i++)
         {
               city = paths[i];
               var latitude = latitudes[city];
               var longitude = longitudes[city];
               distance += (float)Math.Sqrt(Square(latitude - previousLatitude) 
														+ Square(longitude - previousLongitude));
               previousLatitude = latitude;
               previousLongitude = longitude;
         }

         return distance;
      }
   }

	internal class AnswerBetter : Answer {
		protected override int[] RoutePermutation() {
			int cities = Tsp.NumCities;
			int[] path = new int[cities];
			for (var city = 0; city < cities; city++) { path[city] = city; }

			// Credit: SpaceRat. 
			// http://www.daniweb.com/software-development/cpp/code/274075/all-permutations-non-recursive
			var divisor = 1L;
			for (int city = cities; city > 1; ) {
				var dest		= (int) ((Permutation / divisor) % city);
				divisor		*= city;
				city--;

				var swap		= path[dest];
				path[dest]	= path[city];
				path[city]	= swap;
			}
			return path;
		}
	}

   internal static class Cities  {
// original Baker,44.78,117.83 changed to remove duplicate solution for 12 cities.
        public static string List =
@"Albany,42.67,73.75
Albuquerque,35.08,106.65
Amarillo,35.18,101.83
Anchorage,61.22,149.9
Atlanta,33.75,84.38
Austin,30.27,97.73
Baker,44.78,117.83
Baltimore,39.3,76.63
Bangor,44.8,68.78
Birmingham,33.5,86.83
Bismarck,46.8,100.78
Boise,43.6,116.22
Boston,42.35,71.08
Buffalo,42.92,78.83
Carlsbad,32.43,104.25
Charleston,32.78,79.93
Charleston,38.35,81.63
Charlotte,35.23,80.83
Cheyenne,41.15,104.87
Chicago,41.83,87.62
Cincinnati,39.13,84.5
Cleveland,41.47,81.62
Columbia,34,81.03
Columbus,40,83.02
Dallas,32.77,96.77
Denver,39.75,105
Des Moines,41.58,93.62
Detroit,42.33,83.05
Dubuque,42.52,90.67
Duluth,46.82,92.08
Eastport,44.9,67
El Centro,32.63,115.55
El Paso,31.77,106.48
Eugene,44.05,123.08
Fargo,46.87,96.8
Flagstaff,35.22,111.68
Fort Worth,32.72,97.32
Fresno,36.73,119.8
Grand Junction,39.08,108.55
Grand Rapids,42.97,85.67
Havre,48.55,109.72
Helena,46.58,112.03
Honolulu,21.3,157.83
Hot Springs,34.52,93.05
Houston,29.75,95.35
Idaho Falls,43.5,112.02
Indianapolis,39.77,86.17
Jackson,32.33,90.2
Jacksonville,30.37,81.67
Juneau,58.3,134.4
Kansas City,39.1,94.58
Key West,24.55,81.8
Knoxville,35.95,83.93
Las Vegas,36.17,115.2
Lewiston,46.4,117.03
Lincoln,40.83,96.67
Long Beach,33.77,118.18
Los Angeles,34.05,118.25
Louisville,38.25,85.77
Manchester,43,71.5
Memphis,35.15,90.05
Miami,25.77,80.2
Milwaukee,43.03,87.92
Minneapolis,44.98,93.23
Mobile,30.7,88.05
Montgomery,32.35,86.3
Montpelier,44.25,72.53
Nashville,36.17,86.78
Newark,40.73,74.17
New Orleans,29.95,90.07
New York,40.78,73.97
Nome,64.42,165.5
Oakland,37.8,122.27
Oklahoma City,35.43,97.47
Omaha,41.25,95.93
Philadelphia,39.95,75.17
Phoenix,33.48,112.07
Pierre,44.37,100.35
Pittsburgh,40.45,79.95
Portland,43.67,70.25
Portland,45.52,122.68
Providence,41.83,71.4
Raleigh,35.77,78.65
Reno,39.5,119.82
Richfield,38.77,112.08
Richmond,37.55,77.48
Roanoke,37.28,79.95
Sacramento,38.58,121.5
Savannah,32.08,81.08
Seattle,47.62,122.33
Shreveport,32.47,93.7
Sitka,57.17,135.25
Spokane,47.67,117.43
Springfield,39.8,89.63
Springfield,42.1,72.57
Springfield,37.22,93.28
Syracuse,43.03,76.13
Tampa,27.95,82.45
Toledo,41.65,83.55
Tulsa,36.15,95.98
Washington,38.88,77.03
Wichita,37.72,97.28
Wilmington,34.23,77.95";
    }
}
