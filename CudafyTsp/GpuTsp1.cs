#region License - Microsoft Public License - from PG Software Solutions Inc.
/***********************************************************************************
 * This software is copyright © 2012 by PG Software Solutions Inc. and licensed under
 * the Microsoft Public License (http://cudafytuningtutorial.codeplex.com/license).
 * 
 * Author:			Pieter Geerkens
 * Organization:	PG Software Solutions Inc.
 * *********************************************************************************/
#endregion
using System;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace CudafyTuningTsp {
	public class GpuTsp1_SeparateClass : AbstractTsp {
		static GpuTsp1_SeparateClass() {
			AbstractTsp.BuildCityData( (city, @lat, @long) => {
				_latitudes[city]	= @lat;
				_longitudes[city]	= @long;	}
			);
		}

		[Cudafy]
		public struct AnswerStruct { public float distance; public int  pathNo; } 

		internal override Answer GetAnswer() {
			var stopWatchLoad		= Stopwatch.StartNew();
         using (var gpu			= CudafyHost.GetDevice()) {
				gpu.LoadModule(CudafyTranslator.Cudafy());
				LoadTime				= stopWatchLoad.ElapsedMilliseconds;

				var stopWatchRun	= Stopwatch.StartNew();
				var gpuLatitudes	= gpu.CopyToDevice(_latitudes.ToArray());
				var gpuLongitudes	= gpu.CopyToDevice(_longitudes.ToArray());
				var answer			= new AnswerStruct[_blocksPerGrid];;
				var gpuAnswer		= gpu.Allocate(answer);

				gpu.SafeLaunch(_blocksPerGrid, _threadsPerBlock,
					GpuFindPathDistance, _cities, (int)_permutations, gpuLatitudes, gpuLongitudes, gpuAnswer);

				gpu.Synchronize();
				gpu.CopyFromDevice(gpuAnswer, answer);
				gpu.FreeAll();

				var bestDistance		= float.MaxValue;
				var bestPermutation	= 0;
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
			int cities, int permutations, float[] latitudes, float[] longitudes, AnswerStruct[] answer) {

         var threadIndex		= thread.threadIdx.x;	// thread index within the block
         var blockIndex			= thread.blockIdx.x;		// block index within the grid
         var blocksPerGrid		= thread.gridDim.x;
				
         var threadsPerGrid	= thread.blockDim.x * blocksPerGrid;
         var paths				= thread.AllocateShared<int>("path",	_threadsPerBlock, _cities);
         var bestDistances		= thread.AllocateShared<float>("dist", _threadsPerBlock);
         var bestPermutations = thread.AllocateShared<int>("perm",	_threadsPerBlock);

         var permutation		= (threadIndex + blockIndex * thread.blockDim.x);
			var bestDistance		= float.MaxValue;
         var bestPermutation	= 0;
         while (permutation < permutations) {
            var distance = FindPathDistance(cities, permutations, permutation,
									latitudes, longitudes, paths, threadIndex);
            if (distance < bestDistance) {
               bestDistance		= distance;
               bestPermutation	= permutation;
            }
            permutation			  += threadsPerGrid;
         }

         bestDistances[threadIndex]		= bestDistance;
         bestPermutations[threadIndex] = bestPermutation;
         thread.SyncThreads();

			// credit: CUDA By Example, page 79:
			// http://www.amazon.com/CUDA-Example-Introduction-General-Purpose-Programming/dp/0131387685
         for (var i = thread.blockDim.x / 2; i > 0; i /= 2) {
            if (threadIndex < i) {
               if (bestDistances[threadIndex] > bestDistances[threadIndex + i]) {
                  bestDistances[threadIndex]		= bestDistances[threadIndex + i];
                  bestPermutations[threadIndex] = bestPermutations[threadIndex + i];
               }
            }
            thread.SyncThreads();
         }

         if (threadIndex == 0) {
            answer[thread.blockIdx.x].distance	= bestDistances[0];
            answer[thread.blockIdx.x].pathNo		= bestPermutations[0];
         }
      }
      [Cudafy]
      public static float FindPathDistance(int cities, int permutations, int permutation,
			float[] latitudes, float[] longitudes, int[,] paths, int pathIndex) {
         PathFromRoutePermutation(cities, permutations, permutation, paths, pathIndex);

         float distance				= 0;
         var city						= paths[pathIndex, 0];
         var previousLatitude		= latitudes[city];
         var previousLongitude	= longitudes[city];

         for (var i = 1; i < cities; i++) {
            city					= paths[pathIndex, i];
            var latitude		= latitudes[city];
            var longitude		= longitudes[city];
            distance				+= (float)Math.Sqrt(Square(latitude - previousLatitude) 
																+ Square(longitude - previousLongitude));
            previousLatitude	= latitude;
            previousLongitude = longitude;
         }

         return distance;
      }

      [Cudafy]
      public static float Square(float value) { return value * value; }

      [Cudafy]
      public static float PathFromRoutePermutation(int cities, int permutations, 
							int permutation, int[,] paths, int pathIndex) {
         for (var i = 0; i < cities; i++) { paths[pathIndex, i] = i; }

			// Credit: SpaceRat. 
			// http://www.daniweb.com/software-development/cpp/code/274075/all-permutations-non-recursive
         for (int remaining = cities, divisor = permutations; remaining > 0; remaining--) {
            divisor /= remaining;
            var index	= (permutation / divisor) % remaining;
            var swap		= paths[pathIndex, index];
            paths[pathIndex, index]			= paths[pathIndex, remaining-1];
            paths[pathIndex, remaining-1]	= swap;
         }
         return 0;
		}
		#endregion
	}
}
