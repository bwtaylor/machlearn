package com.bwtaylor.perceptron

import breeze.linalg.{LinearAlgebra, DenseMatrix, DenseVector}
import breeze.stats.distributions.{ContinuousDistr, Uniform}
import collection.mutable.ListBuffer
import util.Random

object pla {

  type Perceptron = (DenseVector[Double] => Double)

  def dot(x1 : DenseVector[Double], x2: DenseVector[Double]):Double = { (x2.t * x1).valueAt(0) }
  def dual(x: DenseVector[Double]): Perceptron =  {v: DenseVector[Double] => dot(x,v)}

  def randomPoint(distribution: ContinuousDistr[scala.Double], d: Int): DenseVector[Double] = {
    return DenseVector(distribution.sample(d).toArray)
  }

  def randomPerceptron(distribution: ContinuousDistr[Double], d: Int): Perceptron = {
    val p1 = randomPoint(distribution, d)
    val p2 = randomPoint(distribution, d)
    val pDiff = p2 - p1
    val slope = pDiff(1)/pDiff(0) //denominator 0 with probability 0

    // if pDiff is left of the y-axis, the "left" perpendicular vector is south of the X-axis, otherwise north
    val perpV = if (pDiff(0).signum < 0) DenseVector(slope, -1.0) else DenseVector(-slope,1.0)
    val thresh = dot(p1, perpV) //perpV(0)*p1(0) + perpV(1)*p1(1)

    return { x:DenseVector[Double] => (dot(x, perpV) - thresh).signum }
  }

  class Times(n: Int) {
    def *[A](f: => A) { 1 to n foreach { _ => f } }
  }
  implicit def doTimes(i: Int):Times = new Times(i)


  def sample(N: Int, distribution: ContinuousDistr[scala.Double], d: Int): ListBuffer[DenseVector[Double]] = {
    val data = new ListBuffer[DenseVector[Double]]()
    N * {data += randomPoint(distribution,d)}
    return data
  }

  def h(w: DenseVector[Double], x: DenseVector[Double]): Double = {
    ( dot(x,w).signum - 0.1).signum  // first signum in (-1,0,1), second in (-1, 1) with 0 -> misclassified
  }

  def hList(w: DenseVector[Double], xList: ListBuffer[DenseVector[Double]]): ListBuffer[Double] = {
    for (x <- xList) yield h(w, x)
  }

  def misclassified(guess: ListBuffer[Double], truth: ListBuffer[Double]): IndexedSeq[Int] = {
    for(i <- 0 until guess.length if ( guess(i) != truth(i) ) ) yield i
  }

  def randomIndex(list: IndexedSeq[Int]): Int = {
    val rand = new Random();
    return list(rand.nextInt(list.length));
  }

  //def perceptronLearningAlgorithm(f: Perceptron, sample: ListBuffer[DenseVector[Double]], runs: Int)

  def modelData(f: Perceptron, sample: ListBuffer[DenseVector[Double]]):(ListBuffer[DenseVector[Double]], ListBuffer[Double]) = {
    val x = new ListBuffer[DenseVector[Double]]
    val y = new ListBuffer[Double]
    for (v <- sample) {
      y += f(v)
      val xx =  DenseVector.zeros[Double](v.length + 1)
      xx(0) = 1
      for( i <- 1 to v.length) { xx(i) = v(i-1) }
      x += xx
    }
    (x, y)
  }

  def wNext(w: DenseVector[Double], x: ListBuffer[DenseVector[Double]], y: ListBuffer[Double], hy: ListBuffer[Double]): DenseVector[Double] = {
    val d = x(0).length - 1
    val errors = misclassified(hy, y)
    if (errors.length == 0) return w
    val r = randomIndex(errors) // r in [0,N)
    val yr = h(w, x(r))
    val wNew = DenseVector.zeros[Double](d+1)
    (0 to d).foreach(j => wNew(j) = w(j) - x(r)(j)*yr )
    return wNew
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

  def solveRandomPerceptron(N: Int, distribution: ContinuousDistr[scala.Double], d: Int, maxIterations: Int): Int = {

    val f = randomPerceptron(distribution, d)
    val sampleData = sample(N, distribution, d)
    val (x,y) = modelData(f, sampleData)

    val X = toMatrix(x)
    val yVect = DenseVector[Double](y.toArray)

    var wCurrent: DenseVector[Double] = dag(X)*yVect
    //var wCurrent = DenseVector.zeros[Double](d+1)
    var t = 0
    while ( t < maxIterations && misclassified(hList(wCurrent, x), y).length > 0 ) {
      wCurrent = wNext(wCurrent, x, y, hList(wCurrent, x) )
      t+=1
    }
    return t
  }

  def solveRandomPerceptrons(N: Int, distribution: ContinuousDistr[scala.Double], d: Int, maxIterations: Int, runs: Int): List[Int] = {
    val tList = for (i <- 0 until runs) yield {
      val t = solveRandomPerceptron(N, distribution, d, maxIterations)
      println( i+")SUCCESS!! Converged in "+t+" steps" )
      t
    }
    return tList.toList
  }

  def example1() {
    println( solveRandomPerceptrons(10, Uniform(-1, 1), 2, 500, 1000).sum / 1000.0 )
  }
  def example2() { solveRandomPerceptrons(100, Uniform(-1, 1), 2, 10000, 1000) }

  def main(args: Array[String]) {
    println("Hi!")
    example1()
  }
}


