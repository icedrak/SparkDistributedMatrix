/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.matfast.example

import org.apache.spark.sql.matfast.{Dataset, MatfastSession}
import org.apache.spark.sql.matfast.matrix._
import java.util.Random

object BasicMatrixOps {

  def main(args: Array[String]): Unit = {
    val matfastSession = MatfastSession.builder()
      .master("spark://jupiter22:7077")
      .appName("MatFast"+args(3)+ ":" + args(0)+"x"+args(1)+ "x" +args(2))
      .config("spark.scheduler.mode", "FIFO")
//      .config("spark.rpc.message.maxSize", "1000")
//      .config("spark.network.timeout", "10000000s")
                                     .getOrCreate()


    args.foreach(println)
    val leftRow = args(0).toInt
    val leftCol = args(1).toInt
    val rightCol = args(2).toInt
    val leftsparsity = args(5).toDouble
    val rightsparsity = args(6).toDouble


    if(args(3)=="gnmf")
      GNMF(matfastSession,  leftRow, leftCol, rightCol, leftsparsity, rightsparsity)
    else
      runMatrixSelection(matfastSession,  leftRow, leftCol, rightCol, leftsparsity, rightsparsity)

    matfastSession.stop()
  }

  import scala.reflect.ClassTag
  // scalastyle:off
  implicit def kryoEncoder[A](implicit ct: ClassTag[A]) =
    org.apache.spark.sql.Encoders.kryo[A](ct)
  // scalastyle:on

  private def runMatrixTranspose(spark: MatfastSession): Unit = {
    import spark.implicits._
    val b1 = new DenseMatrix(2, 2, Array[Double](1, 1, 2, 2))
    val b2 = new DenseMatrix(2, 2, Array[Double](2, 2, 3, 3))
    val b3 = new DenseMatrix(2, 2, Array[Double](3, 3, 4, 4))
    val b4 = new DenseMatrix(2, 2, Array[Double](4, 5, 6, 7))
    val s1 = new SparseMatrix(2, 2, Array[Int](0, 1, 2),
      Array[Int](1, 0), Array[Double](4, 2))

    // val seq = Seq((0, 0, b1), (0, 1, b2), (1, 0, b3), (1, 1, b4))
    val seq = Seq(MatrixBlock(0, 2, s1), MatrixBlock(2, 3, b2), MatrixBlock(4, 5, b3), MatrixBlock(6, 7, b4)).toDS()
    import spark.MatfastImplicits._
    seq.t().rdd.foreach{ row =>
      // scalastyle:off
      println(row.get(2).asInstanceOf[MLMatrix])
      // scalastyle:on
    }
  }

  private def runMatrixScalar(spark: MatfastSession): Unit = {
    import spark.implicits._
    val b1 = new DenseMatrix(2, 2, Array[Double](1, 1, 2, 2))
    val s1 = new SparseMatrix(2, 2, Array[Int](0, 1, 2),
      Array[Int](1, 0), Array[Double](4, 2))
    val seq = Seq(MatrixBlock(0, 2, b1), MatrixBlock(1, 3, s1)).toDS()
    import spark.MatfastImplicits._
    seq.power(2).rdd.foreach { row =>
      // scalastyle:off
      println(row.get(2).asInstanceOf[MLMatrix])
      // scalastyle:on
    }
  }

  private def runMatrixElement(spark: MatfastSession): Unit = {
    import spark.implicits._
    val b1 = new DenseMatrix(2, 2, Array[Double](1, 1, 2, 2))
    val b2 = new DenseMatrix(2, 2, Array[Double](2, 2, 3, 3))
    val b3 = new DenseMatrix(2, 2, Array[Double](3, 3, 4, 4))
    val s1 = new SparseMatrix(2, 2, Array[Int](0, 1, 2),
      Array[Int](1, 0), Array[Double](4, 2))
    val seq1 = Seq(MatrixBlock(0, 0, b1), MatrixBlock(1, 1, b2)).toDS()
    val seq2 = Seq(MatrixBlock(0, 0, s1), MatrixBlock(0, 1, b3)).toDS()
    import spark.MatfastImplicits._
    seq1.addElement(4, 4, seq2, 4, 4, 2).rdd.foreach { row =>
      val idx = (row.getInt(0), row.getInt(1))
      // scalastyle:off
      println(idx + ":")
      println(row.get(2).asInstanceOf[MLMatrix])
    }
    println("-----------------")
    // scalastyle:on
    seq1.multiplyElement(4, 4, seq2, 4, 4, 2).rdd.foreach { row =>
      val idx = (row.getInt(0), row.getInt(1))
      // scalastyle:off
      println(idx + ":")
      println(row.get(2).asInstanceOf[MLMatrix])
      // scalastyle:on
    }
  }

  private def runMatrixMultiplication(spark: MatfastSession): Unit = {
    import spark.implicits._
    val b1 = new DenseMatrix(2, 2, Array[Double](1, 1, 2, 2))
    val b2 = new DenseMatrix(2, 2, Array[Double](2, 2, 3, 3))
    val b3 = new DenseMatrix(2, 2, Array[Double](3, 3, 4, 4))
    val b4 = new DenseMatrix(2, 2, Array[Double](4, 5, 6, 7))
    val s1 = new SparseMatrix(2, 2, Array[Int](0, 1, 2),
      Array[Int](1, 0), Array[Double](4, 2))
    val mat1 = Seq(MatrixBlock(0, 0, b1), MatrixBlock(1, 1, b2)).toDS()
    val mat2 = Seq(MatrixBlock(0, 0, b3), MatrixBlock(0, 1, b4), MatrixBlock(1, 1, s1)).toDS()
    import spark.MatfastImplicits._
    mat1.matrixMultiply(4, 4, mat2, 4, 4, 2).rdd.foreach { row =>
      val idx = (row.getInt(0), row.getInt(1))
      // scalastyle:off
      println(idx + ":")
      println(row.get(2).asInstanceOf[MLMatrix])
      // scalastyle:on
    }
  }

  private def GNMF(spark: MatfastSession, leftRow:Int, leftCol:Int, rightRow:Int, leftsparsity:Double, rightsparsity:Double): Unit = {
    val blkSize = 1000
    val rank = 500


    val leftRowBlkNum = leftRow
    val leftColBlkNum = 1



    val rightRowBlkNum = 1
    val rightColBlkNum = rightRow


    val leftRowNum = leftRowBlkNum * blkSize
    val leftColNum = leftColBlkNum * blkSize

    val rightRowNum = rightRowBlkNum * blkSize
    val rightColNum = rightColBlkNum * blkSize

    val leftblkMemorySize = leftsparsity * ((blkSize * blkSize * 8) / (1024 * 1024 * 1024 * 1.0))

    val rightblkMemorySize = rightsparsity * ((blkSize * blkSize * 8) / (1024 * 1024 * 1024 * 1.0))

    val limitNumBlk = Math.ceil(2.0 / leftblkMemorySize).toInt
    val rightlimitNumblk =  Math.ceil(2.0 / rightblkMemorySize).toInt

    val estimatieSize = Math.ceil(leftblkMemorySize).toInt * leftRowBlkNum * leftColBlkNum

    val numPart = leftRowBlkNum * leftColBlkNum

    println(s"number of partition: ${numPart}, the size of block: ${leftblkMemorySize}, the limit number of block in a task: ${limitNumBlk}")


    val leftparts = (leftRowBlkNum*leftColBlkNum)/limitNumBlk

    println(s"left matrix : ${leftparts}")

    import spark.implicits._


    val InitSparsity = leftsparsity

    val initVblkSize = InitSparsity * ((blkSize * blkSize * 8) / (1024 * 1024 * 1024 * 1.0))
    val limitVblk = Math.ceil(2.0 / initVblkSize).toInt
    val Vparts = (leftRowBlkNum * rightRowBlkNum) / limitVblk

    val V = spark.sparkContext.parallelize(for(i <- 0 until leftRowBlkNum; j <- 0 until rightColBlkNum) yield (i, j),
      if(Vparts*10 < 90) 90 else Vparts*10)
      .map(coord =>
          MatrixBlock(coord._1, coord._2, SparseMatrix.sprand(blkSize, blkSize, leftsparsity, new Random))
      ).toDS()





    val rightparts = (rightRowBlkNum*rightColBlkNum)/rightlimitNumblk

    println(s"right matrix: ${rightparts}")

    var H = spark.sparkContext.parallelize(for(i <- 0 until rightRowBlkNum; j <- 0 until rightColBlkNum) yield (i, j),
      if(rightparts*10< 90) 90 else rightparts * 10)
      .map(coord =>
         MatrixBlock(coord._1, coord._2, DenseMatrix.rand(rank, blkSize, new Random))
      ).toDS()

    var W = spark.sparkContext.parallelize(for(i <- 0 until leftRowBlkNum; j <- 0 until leftColBlkNum) yield (i, j),
      if(leftparts*10< 90) 90 else leftparts * 10)
      .map(coord =>
        MatrixBlock(coord._1, coord._2, DenseMatrix.rand(blkSize, rank, new Random))
      ).toDS()

    import spark.MatfastImplicits._
//    println(V.printSchema())
    //
    val resultH = H.multiplyElement(rightRowNum, rightColNum,
      W.t().matrixMultiply(leftColNum, leftRowNum, V, leftRowNum, rightColNum, blkSize), leftColNum,
      rightColNum, blkSize)
      .divideElement(rightRowNum, rightColNum,
        W.t().matrixMultiply(leftColNum, leftRowNum, W, leftRowNum, leftColNum, blkSize)
          .matrixMultiply(leftColNum, leftColNum, H, rightRowNum, rightColNum, blkSize), leftColNum, rightColNum, blkSize)


    val resultW =
      W.multiplyElement(leftRowNum, leftColNum,
        V.matrixMultiply(leftRowNum, rightColNum, H.t(), rightColNum, rightRowNum, blkSize), leftRowNum, rightRowNum, blkSize)
        .divideElement(leftRowNum, leftColNum,
          W.matrixMultiply(leftRowNum, leftColNum,
            H.matrixMultiply(rightRowNum, rightColNum, H.t(), rightColNum, rightRowNum, blkSize)
            , rightRowNum, rightRowNum, blkSize)
          ,leftRowNum, leftColNum, blkSize)
//    val result = V.matrixMultiply(leftRowNum, leftColNum, W, rightRowNum, rightColNum, blkSize)

//    resultH.explain()
//    resultH.explain()

//    resultH.rdd.count()
//    resultH.rdd.count()



        resultH.write.parquet(s"hdfs://10.150.20.163:8020/user/root/Me/H.mtx")
        resultW.write.parquet(s"hdfs://10.150.20.163:8020/user/root/Me/W.mtx")

  }



  /*
   * mat1 has the following structure
   * ---------------
   * | 1  2 |      |
   * | 1  2 |      |
   * ---------------
   * |      | 2  3 |
   * |      | 2  3 |
   * ---------------
   * and mat2 looks like the following
   * ---------------
   * | 3  4 | 4  6 |
   * | 3  4 | 5  7 |
   * ---------------
   * |      | 0  2 |
   * |      | 4  0 |
   * ---------------
   */

  private def runMatrixAggregation(spark: MatfastSession): Unit = {
    import spark.implicits._
    val b1 = new DenseMatrix(2, 2, Array[Double](1, 1, 2, 2))
    val b2 = new DenseMatrix(2, 2, Array[Double](2, 2, 3, 3))
    val b3 = new DenseMatrix(2, 2, Array[Double](3, 3, 4, 4))
    val b4 = new DenseMatrix(2, 2, Array[Double](4, 5, 6, 7))
    val s1 = new SparseMatrix(2, 2, Array[Int](0, 1, 2),
      Array[Int](1, 0), Array[Double](4, 2))
    val mat1 = Seq(MatrixBlock(0, 0, b1), MatrixBlock(1, 1, b2)).toDS()
    val mat2 = Seq(MatrixBlock(0, 0, b3), MatrixBlock(0, 1, b4), MatrixBlock(1, 1, s1)).toDS()

    import spark.MatfastImplicits._

    val mat1_rowsum = mat1.t().rowSum(4, 4)
    mat1_rowsum.rdd.foreach { row =>
      val idx = (row.getInt(0), row.getInt(1))
      // scalastyle:off
      println(idx + ":\n" + row.get(2).asInstanceOf[MLMatrix])
      // scalastyle:on
    }
    val mat2_colsum = mat2.colSum(4, 4)
    mat2_colsum.rdd.foreach { row =>
      val idx = (row.getInt(0), row.getInt(1))
      // scalastyle:off
      println(idx + ":\n" + row.get(2).asInstanceOf[MLMatrix])
      // scalastyle:on
    }

    val product_trace = mat1.matrixMultiply(4, 4, mat2, 4, 4, 2).trace(4, 4)
    product_trace.rdd.foreach { row =>
      val idx = (row.getInt(0), row.getInt(1))
      // scalastyle:off
      println(idx + ":\n" + row.get(2).asInstanceOf[MLMatrix])
      // scalastyle:on
    }
  }

  private def runMatrixProjection(spark: MatfastSession): Unit = {
    import spark.implicits._
    val b1 = new DenseMatrix(2, 2, Array[Double](1, 1, 2, 2))
    val b2 = new DenseMatrix(2, 2, Array[Double](2, 2, 3, 3))
    val b3 = new DenseMatrix(2, 2, Array[Double](3, 3, 4, 4))
    val b4 = new DenseMatrix(2, 2, Array[Double](4, 5, 6, 7))
    val s1 = new SparseMatrix(2, 2, Array[Int](0, 1, 2),
      Array[Int](1, 0), Array[Double](4, 2))
    val mat1 = Seq(MatrixBlock(0, 0, b1), MatrixBlock(1, 1, b2)).toDS()
    val mat2 = Seq(MatrixBlock(0, 0, b3), MatrixBlock(0, 1, b4), MatrixBlock(1, 1, s1)).toDS()

    import spark.MatfastImplicits._

    val mat1_proj_row = mat1.project(4, 4, 2, true, 2)
    mat1_proj_row.rdd.foreach { row =>
      val idx = (row.getInt(0), row.getInt(1))
      // scalastyle:off
      println(idx + ":\n" + row.get(2).asInstanceOf[MLMatrix])
      // scalastyle:on
    }

    val mat2_proj_col = mat2.project(4, 4, 2, false, 3)
    mat2_proj_col.rdd.foreach { row =>
      val idx = (row.getInt(0), row.getInt(1))
      // scalastyle:off
      println(idx + ":\n" + row.get(2).asInstanceOf[MLMatrix])
      // scalastyle:on
    }

    val mat2_X_mat2_col = mat1.matrixMultiply(4, 4, mat2, 4, 4, 2).project(4, 4, 2, false, 3)
    mat2_X_mat2_col.rdd.foreach { row =>
      val idx = (row.getInt(0), row.getInt(1))
      // scalastyle:off
      println(idx + ":\n" + row.get(2).asInstanceOf[MLMatrix])
      // scalastyle:on
    }
  }

  private def runMatrixSelection(spark: MatfastSession, leftRow:Int, leftCol:Int, rightRow:Int, leftsparsity:Double, rightsparsity:Double): Unit = {
    val blkSize = 1000
    val rank = 200

    val leftRowBlkNum = leftRow
    val leftColBlkNum = leftCol



    val rightRowBlkNum = leftColBlkNum
    val rightColBlkNum = rightRow


    val leftRowNum = leftRowBlkNum * blkSize
    val leftColNum = leftColBlkNum * blkSize

    val rightRowNum = rightRowBlkNum * blkSize
    val rightColNum = rightColBlkNum * blkSize

    val leftblkMemorySize = leftsparsity * ((blkSize * blkSize * 8) / (1024 * 1024 * 1024 * 1.0))

    val rightblkMemorySize = rightsparsity * ((blkSize * blkSize * 8) / (1024 * 1024 * 1024 * 1.0))

    val limitNumBlk = Math.ceil(2.0 / leftblkMemorySize).toInt
    val rightlimitNumblk =  Math.ceil(2.0 / rightblkMemorySize).toInt

    val estimatieSize = Math.ceil(leftblkMemorySize).toInt * leftRowBlkNum * leftColBlkNum

    val numPart = leftRowBlkNum * leftColBlkNum

    println(s"number of partition: ${numPart}, the size of block: ${leftblkMemorySize}, the limit number of block in a task: ${limitNumBlk}")


    val leftparts = (leftRowBlkNum*leftColBlkNum)/limitNumBlk

    println(s"left matrix : ${leftparts}")

    import spark.implicits._

    val V = spark.sparkContext.parallelize(for(i <- 0 until leftRowBlkNum; j <- 0 until leftColBlkNum) yield (i, j),
      if(leftparts*10 < 90) 90 else leftparts*10)
      .map(coord =>
        if(leftsparsity == 1) MatrixBlock(coord._1, coord._2, DenseMatrix.rand(blkSize, blkSize, new Random))
        else MatrixBlock(coord._1, coord._2, SparseMatrix.sprand(blkSize, blkSize, leftsparsity, new Random))
      ).toDS()





    val rightparts = (rightRowBlkNum*rightColBlkNum)/rightlimitNumblk

    println(s"right matrix: ${rightparts}")

    val W = spark.sparkContext.parallelize(for(i <- 0 until rightRowBlkNum; j <- 0 until rightColBlkNum) yield (i, j),
      if(rightparts*10< 90) 90 else rightparts * 10)
      .map(coord =>
        if(rightsparsity == 1) MatrixBlock(coord._1, coord._2, DenseMatrix.rand(blkSize, blkSize, new Random))
        else MatrixBlock(coord._1, coord._2, SparseMatrix.sprand(blkSize, blkSize, rightsparsity, new Random))
      ).toDS()

    import spark.MatfastImplicits._
    println(V.printSchema())
    //
    val result = V.matrixMultiply(leftRowNum, leftColNum, W, rightRowNum, rightColNum, blkSize)

    result.explain()

    result.write.parquet(s"hdfs://10.150.20.163:8020/user/root/Me/${leftRowBlkNum}K${rightColBlkNum}K_C.mtx")

  }
}