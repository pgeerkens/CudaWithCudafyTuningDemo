using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace DemonstratePermutations {
	class PermutationsDemo {
		static void Main(string[] args) {
			var cities = 5;
			PrintPermutations(cities,SpaceRat);		Console.WriteLine();
			PrintPermutations(cities,Geerkens);		Console.WriteLine();
			Console.ReadLine();
		}

		public static long Permutations { get; private set; }

		public static void PrintPermutations(int cities, Func<long,int,int[]> action) {
			Permutations = 1;
			for (int i=1; i<=cities; i++) { Permutations *= i; }
			var partition = cities;

			Console.WriteLine("");
			Console.WriteLine("{0}: {1} permutations of {2}",action.Method.Name, Permutations, cities);
			for (int permutation=0; permutation<Permutations; permutation++) {
				if (permutation%partition == 0) Console.WriteLine("");

				var path = action(permutation, cities);
				var s = "";	foreach (int c in path) { s+= string.Format("{0},",c); }

				Console.Write(string.Format("#{1,3}: ({0})  ",s, permutation));
			}
			Console.WriteLine("");
			Console.WriteLine(
@"--------------------------------------------------------------------------------------------------");
		}

      public static int[] Geerkens(long permutation, int cities) {
			int[] path = new int[cities];
         for (var city = 0; city < cities; city++) { path[city] = city; }

			var divisor = 1L;
         for (int city = cities; city > 1; /* decrement in loop body */) {
            var dest		= (int)((permutation / divisor) % city);
				divisor		*= city;

			city--;

            var swap		= path[dest];
            path[dest]	= path[city];
            path[city]	= swap;
         }

         return path;
		}
      public static int[] SpaceRat(long permutation, int cities) {
			int[] path = new int[cities];
         for (var city = 0; city < cities; city++) { path[city] = city; }

			// Credit: SpaceRat. 
			// http://www.daniweb.com/software-development/cpp/code/274075/all-permutations-non-recursive
			var divisor = Permutations;
         for (int city = cities; city > 0; /* decrement in loop body */) {
            divisor		/= city;
            var dest		= (int)((permutation / divisor) % city);

			city--;

            var swap		= path[dest];
            path[dest]	= path[city];
            path[city]	= swap;
         }

         return path;
		}
	}
}
