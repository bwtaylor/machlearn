package com.bwtaylor.perceptron

import breeze.linalg.{LinearAlgebra, DenseMatrix, DenseVector}
import breeze.stats.distributions.{Uniform, ContinuousDistr}
import collection.mutable.ListBuffer

object linearRegression {

  type Perceptron = (DenseVector[Double] => Double)

  def dot(x1: DenseVector[Double], x2: DenseVector[Double]): Double = { (x2.t * x1).valueAt(0) }

  def dual(x: DenseVector[Double]): Perceptron = { v: DenseVector[Double] => dot(x, v) }

  def randomPoint(distribution: ContinuousDistr[scala.Double], d: Int): DenseVector[Double] = {
    return DenseVector(distribution.sample(d).toArray)
  }

  def randomPerceptron(distribution: ContinuousDistr[Double], d: Int): Perceptron = {
    val p1 = randomPoint(distribution, d)
    val p2 = randomPoint(distribution, d)
    val pDiff = p2 - p1
    val slope = pDiff(1) / pDiff(0) //denominator 0 with probability 0

    // if pDiff is left of the y-axis, the "left" perpendicular vector is south of the X-axis, otherwise north
    val perpV = if (pDiff(0).signum < 0) DenseVector(slope, -1.0) else DenseVector(-slope, 1.0)
    val thresh = dot(p1, perpV) //perpV(0)*p1(0) + perpV(1)*p1(1)

    return { x: DenseVector[Double] => (dot(x, perpV) - thresh).signum }
  }

  class Times(n: Int) {
    def *[A](f: => A) { 1 to n foreach { _ => f } }
  }

  implicit def doTimes(i: Int): Times = new Times(i)


  def sample(N: Int, distribution: ContinuousDistr[scala.Double], d: Int): ListBuffer[DenseVector[Double]] = {
    val data = new ListBuffer[DenseVector[Double]]()
    N * { data += randomPoint(distribution, d) }
    return data
  }

  def h(w: DenseVector[Double], x: DenseVector[Double]): Double = {
    (dot(x, w).signum - 0.1).signum // first signum in (-1,0,1), second in (-1, 1) with 0 -> misclassified
  }

  def hList(w: DenseVector[Double], xList: ListBuffer[DenseVector[Double]]): ListBuffer[Double] = {
    for (x <- xList) yield h(w, x)
  }

  def misclassified(guess: ListBuffer[Double], truth: ListBuffer[Double]): IndexedSeq[Int] = {
    for (i <- 0 until guess.length if (guess(i) != truth(i))) yield i
  }

  def modelData(f: Perceptron, sample: ListBuffer[DenseVector[Double]]): (ListBuffer[DenseVector[Double]], ListBuffer[Double]) = {
    val x = new ListBuffer[DenseVector[Double]]
    val y = new ListBuffer[Double]
    for (v <- sample) {
      y += f(v)
      val xx = DenseVector.zeros[Double](v.length + 1)
      xx(0) = 1
      for (i <- 1 to v.length) {
        xx(i) = v(i - 1)
      }
      x += xx
    }
    (x, y)
  }

  def toMatrix(x:ListBuffer[DenseVector[Double]]): DenseMatrix[Double] = {
    val rows = x.length
    val cols = x(0).length
    val m = DenseMatrix.zeros[Double](rows,cols)
    (0 until rows).map(i => m(i,::) := x(i).t )
    return m
  }

  def dag(X:  DenseMatrix[Double]): DenseMatrix[Double]  = {
    LinearAlgebra.inv(X.t * X)*X.t
  }

  def calcProbDiff(w: DenseVector[Double], f: Perceptron):Double = {
    var cnt = 0
    for (x1 <- -1.0 to 1.0 by .01) for( x2 <- -1.0 to 1.0 by .01) {
      val xx = DenseVector[Double](x1,x2)
      val xxx = DenseVector[Double](1,x1,x2)
      if ( h(w,xxx) != f(xx) ) cnt+=1
    }
    return cnt / 40401.0
  }

  def solveRandomPerceptron(N: Int, distribution: ContinuousDistr[scala.Double], d: Int): Double = {
    val f = randomPerceptron(distribution, d)
    val sampleData = sample(N, distribution, d)
    val (x, y) = modelData(f, sampleData)

    val X = toMatrix(x)
    val yVect = DenseVector[Double](y.toArray)
    val w = dag(X)*yVect
    val misses = misclassified(hList(w,x),y)
    val eIn = (1.0 * misses.length) / y.length
    val eOut = calcProbDiff(w,f)
    println(eIn, eOut, misses)
    return eOut
  }

  def solveRandomPerceptrons(N: Int, distribution: ContinuousDistr[scala.Double], d: Int, runs: Int): List[Double] = {
    val listEin = for (i <- 0 until runs) yield solveRandomPerceptron(N, distribution, d)
    return listEin.toList
  }

  def main(args: Array[String]) {
    println(solveRandomPerceptrons(100, Uniform(-1,1), 2, 1000).sum / 1000)
  }

}
