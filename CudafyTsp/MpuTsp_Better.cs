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
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace CudafyTuningTsp {
	public class MpuTsp_Better : AbstractTspCPU {
		static MpuTsp_Better() {
			AbstractTsp.BuildCityData( (city, @lat, @long) => {
				_latLong[city].Latitude		= @lat;
				_latLong[city].Longitude	= @long;	}
			);
		}

		public static readonly	LatLongStruct[] _latLong = new LatLongStruct[_cities];

		private static object locker = new object();
		public struct LatLongStruct {
			public float Latitude;
			public float Longitude;

			public float Distance(LatLongStruct other) {
            return (float)Math.Sqrt(Square(Latitude  - other.Latitude) 
											 + Square(Longitude - other.Longitude));
			}
			public float Square(float value) { return value * value; }
		}
		private struct LocalData {
         public float	BestDistance;
         public long		BestPermutation;
			public LocalData(float bestDistance, long bestPermutation) {
				BestDistance		= bestDistance;
				BestPermutation	= bestPermutation;
			}
		}

      internal override Answer GetAnswer() {
         var stopWatch			= Stopwatch.StartNew();
         var bestDistance		= float.MaxValue;
         var bestPermutation	= -1L;
         var locker				= new Object();

			Parallel.For(0, _permutations, 
				() => new LocalData(float.MaxValue, -1L),
				(permutation, state, localData) => {
					var path			= new int[1, _cities];
					var distance	= FindPathDistance( permutation, path, 0);
					if (distance < localData.BestDistance) {
						localData.BestDistance		= distance;
						localData.BestPermutation	= permutation;
					}
					return localData;
				},
				(localData) => {
					lock (locker) { 
						if (localData.BestDistance < bestDistance) {
							bestDistance		= localData.BestDistance;
							bestPermutation	= localData.BestPermutation;
						}
					}
				}
			);

			return new Answer { 
				Distance		= bestDistance, 
				Permutation	= bestPermutation,
				msLoadTime	= LoadTime, 
				msRunTime	= stopWatch.ElapsedMilliseconds
			};
      }
      public static float FindPathDistance(long  permutation, int[,] paths, int pathIndex) {
         PathFromRoutePermutation(permutation, paths, pathIndex);

         float distance		= 0;
         int city				= paths[pathIndex, 0];
			var prevLatLong	= _latLong[city];

         for (int i = 1; i < _cities; i++) {
            city				= paths[pathIndex, i];
				var latLong		= _latLong[city];
				distance			+= latLong.Distance(prevLatLong);
            prevLatLong		= latLong;
         }

         return distance;
      }
      public static float PathFromRoutePermutation(long  permutation, 
							int[,] paths, int pathIndex) {
         for (int i = 0; i < _cities; i++) { paths[pathIndex, i] = i; }

			// Credit: SpaceRat. 
			// http://www.daniweb.com/software-development/cpp/code/274075/all-permutations-non-recursive
			long  divisor = _permutations;
         for (int remaining = _cities; remaining > 0; remaining--) {
            divisor		/= remaining;
            int index	= (int)((permutation / divisor) % remaining);
            int swap		= paths[pathIndex, index];
            paths[pathIndex, index]			= paths[pathIndex, remaining-1];
            paths[pathIndex, remaining-1]	= swap;
         }
         return 0;
		}
	}
}
