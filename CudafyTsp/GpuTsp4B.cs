using System;
using System.Diagnostics;
using System.Linq;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace CudafyTsp {
	public class GpuTsp4_DivisorsCachedGlobal : AbstractTsp {
		public static readonly LatLongStruct[] _latLong	= new LatLongStruct[_cities];
		static GpuTsp4_DivisorsCachedGlobal() {
			AbstractTsp.BuildCityData( (city, @lat, @long) => {
				_latLong[city].Latitude		= @lat;
				_latLong[city].Longitude	= @long;	}
			);
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

		[Cudafy]
		public static long[] gpuDivisors	= new long[_cities];

		[Cudafy]
		public struct AnswerStruct { public float distance; public long pathNo; } 

		internal override Answer GetAnswer() {
			var stopWatchLoad		= Stopwatch.StartNew();
			using (var gpu			= CudafyHost.GetDevice()) { 
				var arch				= gpu.GetDeviceProperties().Capability.GetArchitecture();
				gpu.LoadModule(CudafyTranslator.Cudafy(ePlatform.x64,arch));
				LoadTime				= stopWatchLoad.ElapsedMilliseconds;

				var stopWatchRun	= Stopwatch.StartNew();
				var gpuLatLong		= gpu.CopyToDevice(_latLong.ToArray());
				var divisors		= new long[_cities];
				long divisor		= _permutations;
				for (int city=_cities; city>0; /* decrement in loop body */) {
					divisor /= city;	
					city--;
					divisors[city] = divisor; 
				}
				gpu.CopyToConstantMemory(divisors,gpuDivisors);

				var answer			= new AnswerStruct[_blocksPerGrid];;
				var gpuAnswer		= gpu.Allocate(answer);

				gpu.SafeLaunch(_blocksPerGrid, _threadsPerBlock,
					GpuFindPathDistance,	_permutations, gpuLatLong, gpuAnswer);

				gpu.Synchronize();
				gpu.CopyFromDevice(gpuAnswer, answer);
				gpu.FreeAll();

				var bestDistance		= float.MaxValue;
				var bestPermutation	= 0L;
				for (var i = 0; i < _blocksPerGrid; i++) {
					if (answer[i].distance < bestDistance) {
						bestDistance		= answer[i].distance;
						bestPermutation	= answer[i].pathNo;
					}
				}

				return new Answer { 
					Distance		= bestDistance, 
					Permutation	= bestPermutation, 
					msLoadTime	= LoadTime, 
					msRunTime	= stopWatchRun.ElapsedMilliseconds
				};
			}
		}

		#region Cudafy
		[Cudafy]
      public static void GpuFindPathDistance(GThread thread, 
			long permutations, LatLongStruct[] gpuLatLong, AnswerStruct[] answer) {

         var threadsPerGrid	= thread.blockDim.x * thread.gridDim.x;
         var path					= thread.AllocateShared<int>("path", _cities, _threadsPerBlock);
         var bestDistances		= thread.AllocateShared<float>("dist", _threadsPerBlock);
         var bestPermutations = thread.AllocateShared<long> ("perm",	_threadsPerBlock);

         var permutation		= (long)(thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x);
			var bestDistance		= float.MaxValue;
         var bestPermutation	= 0L;
         while (permutation < permutations) {
            var distance = FindPathDistance(thread, permutations, permutation, gpuLatLong, path);
            if (distance < bestDistance) {
               bestDistance		= distance;
               bestPermutation	= permutation;
            }
            permutation			  += threadsPerGrid;
         }

         bestDistances[thread.threadIdx.x]		= bestDistance;
         bestPermutations[thread.threadIdx.x]	= bestPermutation;
         thread.SyncThreads();

			// credit: CUDA By Example, page 79:
			// http://www.amazon.com/CUDA-Example-Introduction-General-Purpose-Programming/dp/0131387685
         for (int i = thread.blockDim.x / 2; i > 0; i /= 2) {
            if (thread.threadIdx.x < i) {
               if (bestDistances[thread.threadIdx.x] > bestDistances[thread.threadIdx.x + i]) {
                  bestDistances[thread.threadIdx.x]		= bestDistances[thread.threadIdx.x + i];
                  bestPermutations[thread.threadIdx.x]	= bestPermutations[thread.threadIdx.x + i];
               }
            }
            thread.SyncThreads();
         }

         if (thread.threadIdx.x == 0) {
            answer[thread.blockIdx.x].distance	= bestDistances[0];
            answer[thread.blockIdx.x].pathNo		= bestPermutations[0];
         }
      }
      [Cudafy]
      public static float FindPathDistance(GThread thread, 
				long  permutations, long  permutation, LatLongStruct[] gpuLatLong, int[,] path) {
         PathFromRoutePermutation(thread, permutations, permutation, path);

         float distance		= 0;
         int city				= path[0, thread.threadIdx.x];
			var prevLatLong	= gpuLatLong[city];

         for (int i = 1; i < _cities; i++) {
            city				= path[i, thread.threadIdx.x];
				var latLong		= gpuLatLong[city];
				distance			+= latLong.Distance(prevLatLong);
            prevLatLong		= latLong;
         }

         return distance;
      }

      [Cudafy]
      public static float PathFromRoutePermutation(GThread thread, 
				long  permutations, long  permutation, int[,] path) {
         for (int city = 0; city < _cities; city++) { path[city, thread.threadIdx.x] = city; }

			// Credit: SpaceRat. 
			// http://www.daniweb.com/software-development/cpp/code/274075/all-permutations-non-recursive
         for (int city = _cities; city > 0; /* decrement in loop body */) {
            long divisor	= gpuDivisors[city-1];
            var dest			= (int)(permutation / divisor) % city;
				city--;
            var swap									= path[dest, thread.threadIdx.x];
            path[dest, thread.threadIdx.x]	= path[city, thread.threadIdx.x];
            path[city, thread.threadIdx.x]	= swap;
         }
         return 0;
		}
		#endregion
	}
}
