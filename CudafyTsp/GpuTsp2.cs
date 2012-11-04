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
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace CudafyTuningTsp {
	public class GpuTsp2_StructArray : AbstractTsp {
		public static readonly	LatLongStruct[] _latLong = new LatLongStruct[_cities];
		static GpuTsp2_StructArray() {
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
		public struct AnswerStruct { public float distance; public int  pathNo; } 

		internal override Answer GetAnswer() {
			var stopWatchLoad		= Stopwatch.StartNew();
			using (var gpu			= CudafyHost.GetDevice()) { 
				gpu.LoadModule(CudafyTranslator.Cudafy());
				LoadTime				= stopWatchLoad.ElapsedMilliseconds;

				var stopWatchRun	= Stopwatch.StartNew();
				var gpuLatLong		= gpu.CopyToDevice(_latLong.ToArray());
				var answer			= new AnswerStruct[_blocksPerGrid];;
				var gpuAnswer		= gpu.Allocate(answer);

				gpu.SafeLaunch(_blocksPerGrid, _threadsPerBlock,
					GpuFindPathDistance,	(int)_permutations, gpuLatLong, gpuAnswer);

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
			int permutations, LatLongStruct[] gpuLatLong, AnswerStruct[] answer) {

         var threadsPerGrid	= thread.blockDim.x * thread.gridDim.x;
         var paths				= thread.AllocateShared<int>("path",	_threadsPerBlock, _cities);
         var bestDistances		= thread.AllocateShared<float>("dist", _threadsPerBlock);
         var bestPermutations = thread.AllocateShared<int>("perm",	_threadsPerBlock);

         var permutation		= thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
			var bestDistance		= float.MaxValue;
         var bestPermutation	= 0;
         while (permutation < permutations) {
            var distance = FindPathDistance(permutations, permutation, gpuLatLong, paths, thread.threadIdx.x);
            if (distance < bestDistance) {
               bestDistance		= distance;
               bestPermutation	= permutation;
            }
            permutation			  += threadsPerGrid;
         }

         bestDistances[thread.threadIdx.x]		= bestDistance;
         bestPermutations[thread.threadIdx.x] = bestPermutation;
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
      public static float FindPathDistance(int permutations, int permutation, 
			LatLongStruct[] gpuLatLong, int[,] paths, int pathIndex) {
         PathFromRoutePermutation(permutations, permutation, paths, pathIndex);

         float distance		= 0;
         int city				= paths[pathIndex, 0];
			var prevLatLong	= gpuLatLong[city];

         for (int i = 1; i < _cities; i++) {
            city				= paths[pathIndex, i];
				var latLong		= gpuLatLong[city];
				distance			+= latLong.Distance(prevLatLong);
            prevLatLong		= latLong;
         }

         return distance;
      }

      [Cudafy]
      public static float PathFromRoutePermutation(int permutations, int permutation, 
							int[,] paths, int pathIndex) {
         for (int i = 0; i < _cities; i++) { paths[pathIndex, i] = i; }

			// Credit: SpaceRat. 
			// http://www.daniweb.com/software-development/cpp/code/274075/all-permutations-non-recursive
			int divisor = permutations;
         for (int remaining = _cities; remaining > 0; remaining--) {
            divisor		/= remaining;
            int index	= (permutation / divisor) % remaining;
            int swap		= paths[pathIndex, index];
            paths[pathIndex, index]			= paths[pathIndex, remaining-1];
            paths[pathIndex, remaining-1]	= swap;
         }
         return 0;
		}
		#endregion
	}
}
