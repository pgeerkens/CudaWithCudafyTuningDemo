using System;
using System.Diagnostics;
using System.Dynamic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace CudafyTsp {
   public static class GPGPUExtensions {
		#region  1 through  5
		public static void SafeLaunch<T1>(this GPGPU @this, dim3 gridSize, dim3 blockSize, 
			Action<GThread,T1> action, T1 t1) {
         @this.LaunchAsync(gridSize, blockSize, -1, action.Method.Name, 
            new object[] {t1});
      }

		public static void SafeLaunch<T1,T2>(this GPGPU @this, dim3 gridSize, dim3 blockSize, 
         Action<GThread,T1,T2> action, T1 t1,T2 t2) 
      {
         @this.LaunchAsync(gridSize, blockSize, -1, action.Method.Name, 
            new object[] {t1,t2});
      }

		public static void SafeLaunch<T1,T2,T3>(this GPGPU @this, dim3 gridSize, dim3 blockSize, 
         Action<GThread,T1,T2,T3> action, T1 t1,T2 t2,T3 t3) 
      {
         @this.LaunchAsync(gridSize, blockSize, -1, action.Method.Name, 
            new object[] {t1,t2,t3});
      }

		public static void SafeLaunch<T1,T2,T3,T4>(this GPGPU @this, dim3 gridSize, dim3 blockSize, 
         Action<GThread,T1,T2,T3,T4> action, T1 t1,T2 t2,T3 t3,T4 t4) 
      {
         @this.LaunchAsync(gridSize, blockSize, -1, action.Method.Name, 
            new object[] {t1,t2,t3,t4});
      }

		public static void SafeLaunch<T1,T2,T3,T4,T5>(this GPGPU @this, dim3 gridSize, dim3 blockSize, 
         Action<GThread,T1,T2,T3,T4,T5> action, T1 t1,T2 t2,T3 t3,T4 t4,T5 t5) 
      {
         @this.LaunchAsync(gridSize, blockSize, -1, action.Method.Name, 
            new object[] {t1,t2,t3,t4,t5});
      }
		#endregion

		#region  6 through 10
		public static void SafeLaunch<T1,T2,T3,T4,T5,T6>(this GPGPU @this, dim3 gridSize, dim3 blockSize, 
         Action<GThread,T1,T2,T3,T4,T5,T6> action, T1 t1,T2 t2,T3 t3,T4 t4,T5 t5,T6 t6) 
      {
         @this.LaunchAsync(gridSize, blockSize, -1, action.Method.Name, 
            new object[] {t1,t2,t3,t4,t5,t6});
      }

		public static void SafeLaunch<T1,T2,T3,T4,T5,T6,T7>(this GPGPU @this, dim3 gridSize, dim3 blockSize, 
         Action<GThread,T1,T2,T3,T4,T5,T6,T7> action, T1 t1,T2 t2,T3 t3,T4 t4,T5 t5,T6 t6,T7 t7) 
      {
         @this.LaunchAsync(gridSize, blockSize, -1, action.Method.Name, 
            new object[] {t1,t2,t3,t4,t5,t6,t7});
      }

		public static void SafeLaunch<T1,T2,T3,T4,T5,T6,T7,T8>(this GPGPU @this, dim3 gridSize, dim3 blockSize, 
         Action<GThread,T1,T2,T3,T4,T5,T6,T7,T8> action, T1 t1,T2 t2,T3 t3,T4 t4,T5 t5,T6 t6,T7 t7,T8 t8) 
      {
         @this.LaunchAsync(gridSize, blockSize, -1, action.Method.Name, 
            new object[] {t1,t2,t3,t4,t5,t6,t7,t8});
		}

		public static void SafeLaunch<T1,T2,T3,T4,T5,T6,T7,T8,T9>(this GPGPU @this, dim3 gridSize, dim3 blockSize, 
         Action<GThread,T1,T2,T3,T4,T5,T6,T7,T8,T9> action, T1 t1,T2 t2,T3 t3,T4 t4,T5 t5,T6 t6,T7 t7,T8 t8,T9 t9) 
      {
         @this.LaunchAsync(gridSize, blockSize, -1, action.Method.Name, 
            new object[] {t1,t2,t3,t4,t5,t6,t7,t8,t9});
		}

		public static void SafeLaunch<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10>(this GPGPU @this, dim3 gridSize, dim3 blockSize, 
         Action<GThread,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10> action, T1 t1,T2 t2,T3 t3,T4 t4,T5 t5,T6 t6,T7 t7,T8 t8,T9 t9,T10 t10) 
      {
         @this.LaunchAsync(gridSize, blockSize, -1, action.Method.Name, 
            new object[] {t1,t2,t3,t4,t5,t6,t7,t8,t9,t10});
		}
		#endregion

		#region 11 through 15
		public static void SafeLaunch<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11>
         (this GPGPU @this, dim3 gridSize, dim3 blockSize, 
         Action<GThread,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11> action, 
         T1 t1,T2 t2,T3 t3,T4 t4,T5 t5,T6 t6,T7 t7,T8 t8,T9 t9,T10 t10,T11 t11) 
      {
         @this.LaunchAsync(gridSize, blockSize, -1, action.Method.Name, 
            new object[] {t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11});
		}

		public static void SafeLaunch<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12>
         (this GPGPU @this, dim3 gridSize, dim3 blockSize, 
         Action<GThread,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12> action, 
         T1 t1,T2 t2,T3 t3,T4 t4,T5 t5,T6 t6,T7 t7,T8 t8,T9 t9,T10 t10,T11 t11, T12 t12) 
      {
         @this.LaunchAsync(gridSize, blockSize, -1, action.Method.Name, 
            new object[] {t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12});
		}

		public static void SafeLaunch<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13>
         (this GPGPU @this, dim3 gridSize, dim3 blockSize, 
         Action<GThread,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13> action, 
         T1 t1,T2 t2,T3 t3,T4 t4,T5 t5,T6 t6,T7 t7,T8 t8,T9 t9,T10 t10,T11 t11, T12 t12, T13 t13) 
      {
         @this.LaunchAsync(gridSize, blockSize, -1, action.Method.Name, 
            new object[] {t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13});
		}

		public static void SafeLaunch<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14>
         (this GPGPU @this, dim3 gridSize, dim3 blockSize, 
         Action<GThread,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14> action, 
         T1 t1,T2 t2,T3 t3,T4 t4,T5 t5,T6 t6,T7 t7,T8 t8,T9 t9,T10 t10,T11 t11, T12 t12, T13 t13, T14 t14) 
      {
         @this.LaunchAsync(gridSize, blockSize, -1, action.Method.Name, 
            new object[] {t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14});
		}

		public static void SafeLaunch<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15>
         (this GPGPU @this, dim3 gridSize, dim3 blockSize, 
         Action<GThread,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15> action, 
         T1 t1,T2 t2,T3 t3,T4 t4,T5 t5,T6 t6,T7 t7,T8 t8,T9 t9,T10 t10,T11 t11, T12 t12, T13 t13, T14 t14, T15 t15) 
      {
         @this.LaunchAsync(gridSize, blockSize, -1, action.Method.Name, 
            new object[] {t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15});
		}
		#endregion
	}
}
