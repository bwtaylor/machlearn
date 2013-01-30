package com.bwtaylor

import breeze.linalg.{DenseVector, LinearAlgebra, DenseMatrix}
import breeze.stats.distributions.{Uniform, ContinuousDistr}
import collection.mutable.ListBuffer
import util.Random
import scala.math.pow

object nonlin {

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

  def f(x: DenseVector[Double] ): Double = {
    (pow(x(0),2) + pow(x(1),2) - 0.6).signum
  }

  def solveRandomPerceptron(N: Int, distribution: ContinuousDistr[scala.Double], d: Int): Double = {
    val sampleData = sample(1000, Uniform(-1,1), 2)
    val (x,y0) = modelData(f, sampleData)
    val y = addNoise(y0, 0.1)
    val X = toMatrix(x)
    val yVect = DenseVector[Double](y.toArray)
    val w = dag(X)*yVect
    val misses = misclassified(hList(w,x),y)
    val eIn = (1.0 * misses.length) / y.length
    return eIn
  }

  def solveRandomPerceptrons(N: Int, distribution: ContinuousDistr[scala.Double], d: Int, runs: Int): List[Double] = {
    val tList = for (i <- 0 until runs) yield {
      val  eIn = solveRandomPerceptron(N, distribution, d)
      println( i+") eIn="+eIn )
      eIn
    }
    return tList.toList
  }

  def addNoise(y: ListBuffer[Double], p: Double): ListBuffer[Double] = {
    val rand = new Random()
    val newY = for (yy <- y) yield {
      if (rand.nextDouble() < p) -yy else yy
    }
    return newY
  }

  def example1() {
    println(solveRandomPerceptrons(1000, Uniform(-1,1), 2, 1000).sum / 1000.0)
  }

  def lift(x: DenseVector[Double]): DenseVector[Double] = {
    val x1 = x(1)
    val x2 = x(2)
    return DenseVector[Double](1, x1, x2, x1 * x2, x1*x1, x2*x2)
  }

  def liftList(xList: ListBuffer[DenseVector[Double]]): ListBuffer[DenseVector[Double]] = {
    for (x <- xList) yield lift(x)
  }

  def transToMatrix(xList:ListBuffer[DenseVector[Double]]): DenseMatrix[Double] = {
    val rows = xList.length
    val m = DenseMatrix.zeros[Double](rows,6)
    (0 until rows).map(i => m(i,::) := lift(xList(i)) )
    return m
  }

  def calcProbDiff(w: DenseVector[Double]):Double = {
    var cnt = 0
    for (x1 <- -1.0 to 1.0 by .01) for( x2 <- -1.0 to 1.0 by .01) {
      val xx = DenseVector[Double](x1,x2)
      val xxx = lift(DenseVector[Double](1,x1,x2))
      if ( h(w,xxx) != f(xx) ) cnt+=1
    }
    return cnt / 40401.0
  }


  def nonlinearPerceptron(N: Int, distribution: ContinuousDistr[scala.Double]): Double = {
    val sampleData = sample(1000, Uniform(-1,1), 2)
    val (x,y0) = modelData(f, sampleData)
    val y = addNoise(y0, 0.1)
    val xt = liftList(x)
    val X = toMatrix(liftList(xt))
    val yVect = DenseVector[Double](y.toArray)
    val w = dag(X)*yVect
    println(w)
    //val misses = misclassified(hList(w,xt),y)
    //val eIn = (1.0 * misses.length) / y.length
    //return eIn
    return calcProbDiff(w)

  }

  def example2() {
    val N = 1000
    val distribution = Uniform(-1,1)
    val runs = 1000
    val tList = for (i <- 0 until runs) yield {
      val  error = nonlinearPerceptron(N, distribution)
      println( i+") error="+error )
      error
    }
    println(tList.toList.sum/runs)

  }

  def main(args: Array[String]) {
    println("Hi!")
    example2()
  }
}



