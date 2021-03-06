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

package org.apache.spark.sql.matfast.execution

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkException
import org.apache.spark.sql.catalyst.expressions.{Attribute, GenericInternalRow}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.matfast.matrix._
import org.apache.spark.sql.matfast.partitioner.BlockCyclicPartitioner
import org.apache.spark.sql.matfast.util._

case class ProjectRowDirectExecution(child: SparkPlan,
                                     blkSize: Int,
                                     index: Long) extends MatfastPlan {

  override def output: Seq[Attribute] = child.output

  override def children: Seq[SparkPlan] = child :: Nil

  protected override def doExecute(): RDD[InternalRow] = {
    val rowblkID = (index / blkSize).toInt
    val offset = (index % blkSize).toInt
    val rootRdd = child.execute()
    val rowBlks = rootRdd.map { row =>
      val rid = row.getInt(0)
      val cid = row.getInt(1)
      val matrixInternalRow = row.getStruct(2, 7)
      (rid, (cid, MLMatrixSerializer.deserialize(matrixInternalRow)))
    }.filter(tuple => tuple._1 == rowblkID)
    val rdd = rowBlks.map { tuple =>
      val cid = tuple._2._1
      val matrix = tuple._2._2
      matrix match {
        case den: DenseMatrix =>
          if (!den.isTransposed) { // stored in the column orientation
            val resRow = Array.fill(den.numCols)(0.0)
            for (i <- 0 until resRow.length)  {
              resRow(i) = den.values(offset + i * den.numRows)
            }
            val matBlk = new DenseMatrix(1, resRow.length, resRow)
            ((0, cid), matBlk.asInstanceOf[MLMatrix])
          } else { // stored in the row orientation
            val resRow = Array.fill(den.numRows)(0.0)
            for (i <- 0 until resRow.length) {
              resRow(i) = den.values(offset + i * den.numCols)
            }
            val matBlk = new DenseMatrix(1, resRow.length, resRow)
            ((0, cid), matBlk.asInstanceOf[MLMatrix])
          }
        case sp: SparseMatrix =>
          // Choosing a row in CSC is the same as choosing a column in CSR
          var resValues = ArrayBuffer[Double]()
          var resColPtrs = ArrayBuffer[Int]()
          var cnt = 0
          for (j <- 0 until sp.colPtrs.length - 1) {
            for (k <- 0 until sp.colPtrs(j + 1) - sp.colPtrs(j)) {
              if (offset == sp.rowIndices(k + sp.colPtrs(j))) {
                resValues += sp.values(k + sp.colPtrs(j))
                cnt += 1
              }
            }
            if (j == 0) {
              resColPtrs += 0
            } else {
              resColPtrs += resColPtrs(j - 1) + cnt
            }
          }
          resColPtrs += resValues.length
          val resRowIndices = Array.fill(resValues.length)(offset)
          val matBlk = new SparseMatrix(1, sp.numCols, resColPtrs.toArray,
            resRowIndices, resValues.toArray)
          ((0, cid), matBlk.asInstanceOf[MLMatrix])
        case _ =>
          throw new SparkException("Undefined matrix type in ProjectRowDirectExecute()")
      }
    }
    rdd.map { blk =>
      val res = new GenericInternalRow(3)
      res.setInt(0, blk._1._1)
      res.setInt(1, blk._1._2)
      res.update(2, MLMatrixSerializer.serialize(blk._2))
      res
    }
  }
}

case class ProjectColumnDirectExecution(child: SparkPlan,
                                        blkSize: Int,
                                        index: Long) extends MatfastPlan {

  override def output: Seq[Attribute] = child.output

  override def children: Seq[SparkPlan] = child :: Nil

  protected override def doExecute(): RDD[InternalRow] = {
    val colblkID = (index / blkSize).toInt
    val offset = (index % blkSize).toInt
    val rootRdd = child.execute()
    val colBlks = rootRdd.map { row =>
      val rid = row.getInt(0)
      val cid = row.getInt(1)
      val matrixInternalRow = row.getStruct(2, 7)
      (cid, (rid, MLMatrixSerializer.deserialize(matrixInternalRow)))
    }.filter(tuple => tuple._1  == colblkID)
    val rdd = colBlks.map { tuple =>
      val rid = tuple._2._1
      val matrix = tuple._2._2
      matrix match {
        case den: DenseMatrix =>
          if (!den.isTransposed) { // stored in the column orientation
            val resCol = Array.fill(den.numRows)(0.0)
            for (i <- 0 until resCol.length) {
              resCol(i) = den.values(offset * den.numRows + i)
            }
            val matBlk = new DenseMatrix(resCol.length, 1, resCol)
            ((rid, 0), matBlk.asInstanceOf[MLMatrix])
          } else { // stored in the row orientation
            val resCol = Array.fill(den.numCols)(0.0)
            for (i <- 0 until resCol.length) {
              resCol(i) = den.values(offset * den.numCols + i)
            }
            val matBlk = new DenseMatrix(resCol.length, 1, resCol)
            ((rid, 0), matBlk.asInstanceOf[MLMatrix])
          }
        case sp: SparseMatrix =>
          // Choosing a column in CSC is the same as choosing a row in CSR
          var resValues = ArrayBuffer[Double]()
          var resRowIndices = ArrayBuffer[Int]()
          val resColPtrs = Array[Int](0, sp.colPtrs(offset + 1) - sp.colPtrs(offset))
          for (i <- 0 until sp.colPtrs(offset + 1) - sp.colPtrs(offset)) {
            val k = i + sp.colPtrs(offset)
            resValues += sp.values(k)
            resRowIndices += sp.rowIndices(k)
          }
          val matBlk = new SparseMatrix(sp.numRows, 1, resColPtrs,
            resRowIndices.toArray, resValues.toArray)
          ((rid, 0), matBlk.asInstanceOf[MLMatrix])
        case _ =>
          throw new SparkException("Undefined matrix type in ProjectColumnDirectExecute()")
      }
    }
    rdd.map { blk =>
      val res = new GenericInternalRow(3)
      res.setInt(0, blk._1._1)
      res.setInt(1, blk._1._2)
      res.update(2, MLMatrixSerializer.serialize(blk._2))
      res
    }
  }
}

case class SelectDirectExecution(child: SparkPlan,
                                 blkSize: Int,
                                 rowIdx: Long,
                                 colIdx: Long) extends MatfastPlan {

  override def output: Seq[Attribute] = child.output

  override def children: Seq[SparkPlan] = child :: Nil

  protected override def doExecute(): RDD[InternalRow] = {
    val rowblkID = (rowIdx / blkSize).toInt
    val colblkID = (colIdx / blkSize).toInt
    val rowOffset = (rowIdx % blkSize).toInt
    val colOffset = (colIdx % blkSize).toInt
    val rootRdd = child.execute()
    val blk = rootRdd.map { row =>
      val rid = row.getInt(0)
      val cid = row.getInt(1)
      val matrixInternalRow = row.getStruct(2, 7)
      ((rid, cid), MLMatrixSerializer.deserialize(matrixInternalRow))
    }.filter(tuple => tuple._1 == (rowblkID, colblkID))
    val rdd = blk.map { tuple =>
      val matrix = tuple._2
      matrix match {
        case den: DenseMatrix =>
          val matBlk = new DenseMatrix(1, 1, Array(den.apply(rowOffset, colOffset)))
          ((0, 0), matBlk.asInstanceOf[MLMatrix])
        case sp: SparseMatrix =>
          val matBlk = new DenseMatrix(1, 1, Array(sp.apply(rowOffset, colOffset)))
          ((0, 0), matBlk.asInstanceOf[MLMatrix])
        case _ =>
          throw new SparkException("Undefined matrix type in SelectDirectExecute()")
      }
    }
    rdd.map { blk =>
      val res = new GenericInternalRow(3)
      res.setInt(0, blk._1._1)
      res.setInt(1, blk._1._2)
      res.update(2, MLMatrixSerializer.serialize(blk._2))
      res
    }
  }
}

case class MatrixTransposeExecution(child: SparkPlan) extends MatfastPlan {

  override def output: Seq[Attribute] = child.output

  override def children: Seq[SparkPlan] = child :: Nil

  protected override def doExecute(): RDD[InternalRow] = {
    val rootRdd = child.execute()
    rootRdd.map { row =>
      val rid = row.getInt(0)
      val cid = row.getInt(1)
      val matrixInternalRow = row.getStruct(2, 7)
      val res = new GenericInternalRow(3)
      val matrix = MLMatrixSerializer.deserialize(matrixInternalRow)
      val matrixRow = MLMatrixSerializer.serialize(matrix.transpose)
      res.setInt(0, cid)
      res.setInt(1, rid)
      res.update(2, matrixRow)
      res
    }
  }
}

// this class computes rowSum() on a matrix input
case class RowSumDirectExecution(child: SparkPlan) extends MatfastPlan {

  override def output: Seq[Attribute] = child.output

  override def children: Seq[SparkPlan] = child :: Nil

  protected override def doExecute(): RDD[InternalRow] = {
    val rootRdd = child.execute()
    val rdd = rootRdd.map { row =>
      val rid = row.getInt(0)
      val cid = row.getInt(1)
      val matrixInternalRow = row.getStruct(2, 7)
      val matrix = MLMatrixSerializer.deserialize(matrixInternalRow)
      val colVec = matrix match {
        case den: DenseMatrix =>
          if (!den.isTransposed) { // row sum
            val m = den.numRows
            val arr = new Array[Double](m)
            val values = den.values
            for (i <- 0 until values.length) {
              arr(i % m) += values(i)
            }
            new DenseMatrix(m, 1, arr)
          } else { // column sum
            val n = den.numCols
            val arr = new Array[Double](n)
            val values = den.values
            for (i <- 0 until n) {
              for (j <- 0 until den.numRows) {
                arr(i) += values(i * den.numRows + j)
              }
            }
            new DenseMatrix(n, 1, arr)
          }
        case sp: SparseMatrix =>
          if (!sp.isTransposed) { // CSC format
            val arr = new Array[Double](sp.numRows)
            for (i <- 0 until sp.rowIndices.length) {
              arr(sp.rowIndices(i)) += sp.values(i)
            }
            new DenseMatrix(sp.numRows, 1, arr)
          } else { // CSR format
            val arr = new Array[Double](sp.numCols)
            val colIdx = sp.rowIndices
            for (i <- 0 until colIdx.length) {
              arr(colIdx(i)) += sp.values(i)
            }
            new DenseMatrix(sp.numCols, 1, arr)
          }
        case _ => throw new SparkException("Undefined matrix type in RowSumDirectExecute()")
      }
      (rid, colVec.asInstanceOf[MLMatrix])
    }.reduceByKey(LocalMatrix.add(_, _))
    rdd.map { blk =>
      val rid = blk._1
      val res = new GenericInternalRow(3)
      res.setInt(0, rid)
      res.setInt(1, 0)
      res.update(2, MLMatrixSerializer.serialize(blk._2))
      res
    }
  }
}

case class ColumnSumDirectExecution(child: SparkPlan) extends MatfastPlan {

  override def output: Seq[Attribute] = child.output

  override def children: Seq[SparkPlan] = child :: Nil

  protected override def doExecute(): RDD[InternalRow] = {
    val rootRdd = child.execute()
    val rdd = rootRdd.map { row =>
      val rid = row.getInt(0)
      val cid = row.getInt(1)
      val matrixInternalRow = row.getStruct(2, 7)
      val matrix = MLMatrixSerializer.deserialize(matrixInternalRow)
      val rowVec = matrix match {
        case den: DenseMatrix =>
          if (!den.isTransposed) {
            val n = den.numCols
            val arr = new Array[Double](n)
            val values = den.values
            for (i <- 0 until n) {
              for (j <- 0 until den.numRows) {
                arr(i) += values(i*den.numRows + j)
              }
            }
            new DenseMatrix(1, n, arr)
          } else {
            val m = den.numRows
            val arr = new Array[Double](m)
            val values = den.values
            for (i <- 0 until values.length) {
              arr(i % m) += values(i)
            }
            new DenseMatrix(1, m, arr)
          }
        case sp: SparseMatrix =>
          if (!sp.isTransposed) { // CSC format
            val arr = new Array[Double](sp.numCols)
            for (i <- 0 until sp.colPtrs.length - 1) {
              for (j <- 0 until sp.colPtrs(i + 1) - sp.colPtrs(i)) {
                arr(i) += sp.values(i + j)
              }
            }
            new DenseMatrix(1, sp.numCols, arr)
          } else { // CSR format
            val arr = new Array[Double](sp.numRows)
            val colIdx = sp.rowIndices
            val rowPtrs = sp.colPtrs
            for (i <- 0 until rowPtrs.length - 1) {
              for (j <- 0 until rowPtrs(i + 1) - rowPtrs(i)) {
                arr(i) += sp.values(i + j)
              }
            }
            new DenseMatrix(1, sp.numRows, arr)
          }
        case _ => throw new SparkException("Undefined matrix type in ColumnSumDirectExecute()")
      }
      (cid, rowVec.asInstanceOf[MLMatrix])
    }.reduceByKey(LocalMatrix.add(_, _))
    rdd.map { blk =>
      val cid = blk._1
      val res = new GenericInternalRow(3)
      res.setInt(0, 0)
      res.setInt(1, cid)
      res.update(2, MLMatrixSerializer.serialize(blk._2))
      res
    }
  }
}

case class SumDirectExecution(child: SparkPlan) extends MatfastPlan {

  override def output: Seq[Attribute] = child.output

  override def children: Seq[SparkPlan] = child :: Nil

  protected override def doExecute(): RDD[InternalRow] = {
    val rootRdd = child.execute()
    val rdd = rootRdd.map { row =>
      val matrixInternalRow = row.getStruct(2, 7)
      val matrix = MLMatrixSerializer.deserialize(matrixInternalRow)
      val scalar = matrix match {
        case den: DenseMatrix =>
          new DenseMatrix(1, 1, Array[Double](den.values.sum))
        case sp: SparseMatrix =>
          new DenseMatrix(1, 1, Array[Double](sp.values.sum))
        case _ =>
          throw new SparkException("Undefined matrix type in SumDirectExecute()")
      }
      (0, scalar.asInstanceOf[MLMatrix])
    }.reduceByKey(LocalMatrix.add(_, _))
    rdd.map { blk =>
      val res = new GenericInternalRow(3)
      res.setInt(0, 0)
      res.setInt(1, 0)
      res.update(2, MLMatrixSerializer.serialize(blk._2))
      res
    }
  }
}


case class TraceDirectExecution(child: SparkPlan) extends MatfastPlan {

  override def output: Seq[Attribute] = child.output

  override def children: Seq[SparkPlan] = child :: Nil

  protected override def doExecute(): RDD[InternalRow] = {
    val rootRdd = child.execute()
    val rdd = rootRdd.map { row =>
      val rid = row.getInt(0)
      val cid = row.getInt(1)
      val matrixInternalRow = row.getStruct(2, 7)
      val matrix = MLMatrixSerializer.deserialize(matrixInternalRow)
      ((rid, cid), matrix)
    }
    val traceRdd = rdd.filter(tuple => tuple._1._1 == tuple._1._2).map { row =>
      val localTrace = row._2 match {
        case den: DenseMatrix =>
          val rowNum = den.numRows
          val colNum = den.numCols
          require(rowNum == colNum, s"block is not square, row_num=$rowNum, col_num=$colNum")
          val values = den.values
          // trace is invariant under the transpose operation
          // just compute in a uniform way
          var tr = 0.0
          for (j <- 0 until colNum) {
            tr += values(j * colNum + j)
          }
          val trMat = new DenseMatrix(1, 1, Array[Double](tr))
          (0, trMat.asInstanceOf[MLMatrix])
        case sp: SparseMatrix =>
          // similar to the dense case, no need to distinguish CSC or CSR format
          var tr = 0.0
          val values = sp.values
          val rowIndices = sp.rowIndices
          val colPtrs = sp.colPtrs
          for (j <- 0 until sp.numCols) {
            for (k <- 0 until colPtrs(j + 1) - colPtrs(j)) {
              if (rowIndices(k + colPtrs(j)) == j) {
                tr += values(k + colPtrs(j))
              }
            }
          }
          val trMat = new DenseMatrix(1, 1, Array[Double](tr))
          (0, trMat.asInstanceOf[MLMatrix])
        case _ =>
          throw new SparkException("Undefined matrix type in TraceDirectExecute()")
      }
      localTrace
    }.reduceByKey(LocalMatrix.add(_, _))

    traceRdd.map { blk =>
      val res = new GenericInternalRow(3)
      res.setInt(0, 0)
      res.setInt(1, 0)
      res.update(2, MLMatrixSerializer.serialize(blk._2))
      res
    }
  }
}

case class MatrixScalarAddExecution(child: SparkPlan, alpha: Double) extends MatfastPlan {

  override def output: Seq[Attribute] = child.output

  override def children: Seq[SparkPlan] = child :: Nil

  protected override def doExecute(): RDD[InternalRow] = {
    val rootRdd = child.execute()
    rootRdd.map { row =>
      val rid = row.getInt(0)
      val cid = row.getInt(1)
      val matrixInternalRow = row.getStruct(2, 7)
      val res = new GenericInternalRow(3)
      val matrix = MLMatrixSerializer.deserialize(matrixInternalRow)
      val matrixRow = MLMatrixSerializer.serialize(LocalMatrix.addScalar(matrix, alpha))
      res.setInt(0, rid)
      res.setInt(1, cid)
      res.update(2, matrixRow)
      res
    }
  }
}

case class MatrixScalarMultiplyExecution(child: SparkPlan, alpha: Double) extends MatfastPlan {

  override def output: Seq[Attribute] = child.output

  override def children: Seq[SparkPlan] = child :: Nil

  protected override def doExecute(): RDD[InternalRow] = {
    val rootRdd = child.execute()
    rootRdd.map { row =>
      val rid = row.getInt(0)
      val cid = row.getInt(1)
      val matrixInternalRow = row.getStruct(2, 7)
      val res = new GenericInternalRow(3)
      val matrix = MLMatrixSerializer.deserialize(matrixInternalRow)
      val matrixRow = MLMatrixSerializer.serialize(LocalMatrix.multiplyScalar(alpha, matrix))
      res.setInt(0, rid)
      res.setInt(1, cid)
      res.update(2, matrixRow)
      res
    }
  }
}

case class MatrixPowerExecution(child: SparkPlan, alpha: Double) extends MatfastPlan {

  override def output: Seq[Attribute] = child.output

  override def children: Seq[SparkPlan] = child :: Nil

  protected override def doExecute(): RDD[InternalRow] = {
    val rootRdd = child.execute()
    rootRdd.map { row =>
      val rid = row.getInt(0)
      val cid = row.getInt(1)
      val matrixInternalRow = row.getStruct(2, 7)
      val res = new GenericInternalRow(3)
      val matrix = MLMatrixSerializer.deserialize(matrixInternalRow)
      val matrixRow = MLMatrixSerializer.serialize(LocalMatrix.matrixPow(matrix, alpha))
      res.setInt(0, rid)
      res.setInt(1, cid)
      res.update(2, matrixRow)
      res
    }
  }
}

case class VectorizeExecution(child: SparkPlan,
                              nrows: Long, ncols: Long, blkSize: Int) extends MatfastPlan {

  override def output: Seq[Attribute] = child.output

  override def children: Seq[SparkPlan] = child :: Nil

  protected override def doExecute(): RDD[InternalRow] = {
    val rdd = child.execute()
    val ROW_BLK_NUM = math.ceil(nrows * 1.0 / blkSize).toInt
    rdd.flatMap { row =>
      val i = row.getInt(0)
      val j = row.getInt(1)
      val matrix = MLMatrixSerializer.deserialize(row.getStruct(2, 7))
      val arr = matrix.toArray
      val numLocalRows = matrix.numRows
      val numLocalCols = matrix.numCols
      val buffer = ArrayBuffer[((Int, Int), MLMatrix)]()
      for (t <- 0 until numLocalCols) {
        val key = (j * ROW_BLK_NUM * blkSize + t * ROW_BLK_NUM + i, 0)
        val vecArray = new Array[Double](numLocalRows)
        for (k <- 0 until numLocalRows) {
          vecArray(k) = arr(t * numLocalCols + k)
        }
        buffer.append((key, new DenseMatrix(vecArray.length, 1, vecArray)))
      }
      buffer
    }.map { row =>
      val res = new GenericInternalRow(3)
      res.setInt(0, row._1._1)
      res.setInt(1, row._1._2)
      res.update(2, MLMatrixSerializer.serialize(row._2))
      res
    }
  }
}

case class MatrixElementAddExecution(left: SparkPlan,
                                     leftRowNum: Long,
                                     leftColNum: Long,
                                     right: SparkPlan,
                                     rightRowNum: Long,
                                     rightColNum: Long,
                                     blkSize: Int) extends MatfastPlan {

  override def output: Seq[Attribute] = left.output

  override def children: Seq[SparkPlan] = Seq(left, right)

  protected override def doExecute(): RDD[InternalRow] = {
    require(leftRowNum == rightRowNum, s"Row number not match, " +
      s"leftRowNum = $leftRowNum, rightRowNum = $rightRowNum")
    require(leftColNum == rightColNum, s"Col number not match, " +
      s"leftColNum = $leftColNum, rightColNum = $rightColNum")
    val rdd1 = left.execute()
    val rdd2 = right.execute()
    if (rdd1.partitioner != None) {
      val part = rdd1.partitioner.get
      MatfastExecutionHelper.addWithPartitioner(
        MatfastExecutionHelper.repartitionWithTargetPartitioner(part, rdd1),
        MatfastExecutionHelper.repartitionWithTargetPartitioner(part, rdd2))
    } else if (rdd2.partitioner != None) {
      val part = rdd2.partitioner.get
      MatfastExecutionHelper.addWithPartitioner(
        MatfastExecutionHelper.repartitionWithTargetPartitioner(part, rdd1),
        MatfastExecutionHelper.repartitionWithTargetPartitioner(part, rdd2))
    } else {
      val params = MatfastExecutionHelper.genBlockCyclicPartitioner(leftRowNum, leftColNum, blkSize)
      val part = new BlockCyclicPartitioner(params._1, params._2, params._3, params._4)
      MatfastExecutionHelper.addWithPartitioner(
        MatfastExecutionHelper.repartitionWithTargetPartitioner(part, rdd1),
        MatfastExecutionHelper.repartitionWithTargetPartitioner(part, rdd2))
    }
  }
}

case class MatrixElementMultiplyExecution(left: SparkPlan,
                                          leftRowNum: Long,
                                          leftColNum: Long,
                                          right: SparkPlan,
                                          rightRowNum: Long,
                                          rightColNum: Long,
                                          blkSize: Int) extends MatfastPlan {

  override def output: Seq[Attribute] = left.output

  override def children: Seq[SparkPlan] = Seq(left, right)

  protected override def doExecute(): RDD[InternalRow] = {
    require(leftRowNum == rightRowNum, s"Row number not match, " +
      s"leftRowNum = $leftRowNum, rightRowNum = $rightRowNum")
    require(leftColNum == rightColNum, s"Col number not match, " +
      s"leftColNum = $leftColNum, rightColNum = $rightColNum")
    val rdd1 = left.execute()
    val rdd2 = right.execute()
    if (rdd1.partitioner != None) {
      val part = rdd1.partitioner.get
      MatfastExecutionHelper.multiplyWithPartitioner(
        MatfastExecutionHelper.repartitionWithTargetPartitioner(part, rdd1),
        MatfastExecutionHelper.repartitionWithTargetPartitioner(part, rdd2))
    } else if (rdd2.partitioner != None) {
      val part = rdd2.partitioner.get
      MatfastExecutionHelper.multiplyWithPartitioner(
        MatfastExecutionHelper.repartitionWithTargetPartitioner(part, rdd1),
        MatfastExecutionHelper.repartitionWithTargetPartitioner(part, rdd2))
    } else {
      val params = MatfastExecutionHelper.genBlockCyclicPartitioner(leftRowNum, leftColNum, blkSize)
      val part = new BlockCyclicPartitioner(params._1, params._2, params._3, params._4)
      MatfastExecutionHelper.multiplyWithPartitioner(
        MatfastExecutionHelper.repartitionWithTargetPartitioner(part, rdd1),
        MatfastExecutionHelper.repartitionWithTargetPartitioner(part, rdd2))
    }
  }
}

case class MatrixElementDivideExecution(left: SparkPlan,
                                        leftRowNum: Long,
                                        leftColNum: Long,
                                        right: SparkPlan,
                                        rightRowNum: Long,
                                        rightColNum: Long,
                                        blkSize: Int) extends MatfastPlan {

  override def output: Seq[Attribute] = left.output

  override def children: Seq[SparkPlan] = Seq(left, right)

  protected override def doExecute(): RDD[InternalRow] = {
    require(leftRowNum == rightRowNum, s"Row number not match, " +
      s"leftRowNum = $leftRowNum, rightRowNum = $rightRowNum")
    require(leftColNum == rightColNum, s"Col number not match, " +
      s"leftColNum = $leftColNum, rightColNum = $rightColNum")
    val rdd1 = left.execute()
    val rdd2 = right.execute()
    if (rdd1.partitioner != None) {
      val part = rdd1.partitioner.get
      MatfastExecutionHelper.divideWithPartitioner(
        MatfastExecutionHelper.repartitionWithTargetPartitioner(part, rdd1),
        MatfastExecutionHelper.repartitionWithTargetPartitioner(part, rdd2))
    } else if (rdd2.partitioner != None) {
      val part = rdd2.partitioner.get
      MatfastExecutionHelper.divideWithPartitioner(
        MatfastExecutionHelper.repartitionWithTargetPartitioner(part, rdd1),
        MatfastExecutionHelper.repartitionWithTargetPartitioner(part, rdd2))
    } else {
      val params = MatfastExecutionHelper.genBlockCyclicPartitioner(leftRowNum, leftColNum, blkSize)
      val part = new BlockCyclicPartitioner(params._1, params._2, params._3, params._4)
      MatfastExecutionHelper.divideWithPartitioner(
        MatfastExecutionHelper.repartitionWithTargetPartitioner(part, rdd1),
        MatfastExecutionHelper.repartitionWithTargetPartitioner(part, rdd2))
    }
  }
}

case class MatrixMatrixMultiplicationExecution(left: SparkPlan,
                                               leftRowNum: Long,
                                               leftColNum: Long,
                                               right: SparkPlan,
                                               rightRowNum: Long,
                                               rightColNum: Long,
                                               blkSize: Int) extends MatfastPlan {

  override def output: Seq[Attribute] = left.output

  override def children: Seq[SparkPlan] = Seq(left, right)

  protected override def doExecute(): RDD[InternalRow] = {
    // check for multiplication possibility
    require(leftColNum == rightRowNum, s"Matrix dimension not match, " +
      s"leftColNum = $leftColNum, rightRowNum = $rightRowNum")
    // estimate memory usage
    val memoryUsage = leftRowNum * rightColNum * 8 / (1024 * 1024 * 1024) * 1.0
    if (memoryUsage > 10) {
      // scalastyle:off
      println(s"Caution: matrix multiplication result size = $memoryUsage GB")
      // scalastyle:on
    }
    // compute number of row/col blocks for invoking special matrix multiplication procedure
    val leftColBlkNum = math.ceil(leftColNum * 1.0 / blkSize).toInt
    val rightRowBlkNum = math.ceil(rightRowNum * 1.0 / blkSize).toInt
    if (leftColBlkNum == 1 && rightRowBlkNum == 1) {
      val leftRowBlkNum = leftRowNum / blkSize
      val rightColBlkNum = rightColNum / blkSize
      if (leftRowBlkNum <= rightColBlkNum) {
        println(s"BMM")
        MatfastExecutionHelper.multiplyOuterProductDuplicateLeft(left.execute(), right.execute())
      } else {
        println(s"BMM")
        MatfastExecutionHelper.multiplyOuterProductDuplicateRight(left.execute(), right.execute())
      }
    } else {
      println(s"CPMM")
      MatfastExecutionHelper.matrixMultiplyGeneral(left.execute(), right.execute())
    }
  }
}

case class RankOneUpdateExecution(left: SparkPlan,
                                  leftRowNum: Long,
                                  leftColNum: Long,
                                  right: SparkPlan,
                                  rightRowNum: Long,
                                  rightColNum: Long,
                                  blkSize: Int) extends MatfastPlan {

  override def output: Seq[Attribute] = left.output

  override def children: Seq[SparkPlan] = Seq(left, right)

  protected override def doExecute(): RDD[InternalRow] = {
    require(rightRowNum == 1, s"Vector column size is not 1, but #cols = $rightRowNum")
    require(leftRowNum == rightRowNum, s"Dimension not match for matrix addition, " +
      s"A.nrows = $leftRowNum, " +
    s"A.ncols = ${leftColNum}, B.nrows = $rightRowNum, B.ncols = $rightColNum")
    MatfastExecutionHelper.matrixRankOneUpdate(left.execute(), right.execute())
  }
}