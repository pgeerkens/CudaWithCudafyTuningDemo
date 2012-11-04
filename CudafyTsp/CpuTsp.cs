using System;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace CudafyTsp {
	public class CpuTsp : AbstractTspCPU {
      internal override Answer GetAnswer() {
         var bestDistance		= float.MaxValue;
         var bestPermutation	= -1;
         var stopWatch			= Stopwatch.StartNew();

         for (var permutation = 0; permutation < _permutations; permutation++) {
            var path				= new int[1, _cities];
            var distance		= FindPathDistance(_permutations, permutation, 
					_cities, _latitudes, _longitudes, path, 0);
            if (distance < bestDistance) {
               bestDistance		= distance;
               bestPermutation	= permutation;
            }
         }

         return new Answer { 
				Distance		= bestDistance, 
				Permutation	= bestPermutation,
				msLoadTime	= LoadTime,
				msRunTime	= stopWatch.ElapsedMilliseconds
			};
      }
	}
	public class MpuTsp : AbstractTspCPU {
      internal override Answer GetAnswer() {
         var bestDistance		= float.MaxValue;
         var bestPermutation	= -1L;
         var locker				= new Object();
         var stopWatch			= Stopwatch.StartNew();

         Parallel.For(0, _permutations, permutation => {
            var path				= new int[1, _cities];
            var distance		= FindPathDistance(_permutations, permutation, 
					_cities, _latitudes, _longitudes, path, 0);
            lock (locker) {
               if (distance < bestDistance) {
                  bestDistance		= distance;
                  bestPermutation	= permutation;
               }
            }
         });

         return new Answer { 
				Distance		= bestDistance, 
				Permutation	= bestPermutation,
				msLoadTime	= LoadTime, 
				msRunTime	= stopWatch.ElapsedMilliseconds
			};
      }
	}
}
