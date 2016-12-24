package org.mrb.som

import org.mrb.som._
import scala.math._
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Point

import org.opencv.core.Size
import org.opencv.core.Core
import org.opencv.imgproc.Imgproc

import javax.json._
import java.io.OutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.FileReader

object Test4D {
  var opts: Map[String,(String,String)] = null
  
  var radius: Int = 25
  var rows: Int = 50
  var columns: Int = 50
  var dimension: Int = 4
  var randomVecs: Int = 100
  var epochs = 1
  var its = 500
  var jitter = 0.1
  var corner = false;
  
  var baseVectors: Int = 8
  var randomCounts: Array[Int] = null;
  var start: Double = -20.0
  var end: Double = 20.0
  var step: Double = 2.0
  
  var img: Mat = null
  var l: SomLattice = null
  var ivecs: InputVectors = null
  var theta: TrainingFunction = null
  var progress: Progress = null
  var createUmatrix: Boolean = true
  var umatrix: Mat = null
  var render: UMatrixRender = null
  var displayLearning: Boolean = false
  var displayTime: Boolean = false
  var displayLattice: Boolean = false
   
  var basePath: String = "~/"
  var fileTemplate: String = "test-%E-%T.jpg"
  var showMarkers: Boolean = false
  var time: String = null
  var learn: String = null

  var config: JsonObject = null

  def main(args: Array[String]) = {    
    var mm: (Double, Double) = null // minimum and maximum distances
    var opts: Map[String, (String,String)] = null
    val textMarg = 20        // duplicate value in Progress class below
    var maxRow: Int = 0
    var maxCol: Int = 0
    
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    var baseline: Array[Int] = Array(0)
    val textSize: Size = Imgproc.getTextSize("EPOCH: 0  STEP: 0", Core.FONT_HERSHEY_SIMPLEX, 2, 3, baseline)
    // TODO:
    var fp: File = null
    var os: OutputStream = null
    var o: JsonObject = null
    var jw: JsonWriter = null

    println("Starting 4D project")
    
    opts = cmdLineOptions()
    parseCmdLine(args, opts)
    
//    l.show(false)
    // TODO: only if not set in config
    if (ivecs == null) ivecs = initTrainingVectors(randomVecs, jitter)

    maxRow = l.members.flatten.maxBy[Double](_.y).y.toInt + 1
    maxCol = l.members.flatten.maxBy[Double](_.x).x.toInt + 1
    img = new Mat(2*(maxRow+1)*radius + textSize.height.toInt + 2*textMarg, 2*(maxCol+1)*radius, CvType.CV_8UC3, new Scalar(0,0,0))
    // TODO make number of images configurable
    progress = new Progress(0, 10, radius, img, basePath + fileTemplate, if (showMarkers) ivecs else null)

    if (displayLearning) showLearningFunction()
    if (displayTime) showTimeFunction()
    
    println("Fixed neighborhood functions")
    println("Training for " + epochs + " epochs with " + its + " iterations per cycle")
    println("Network has " + (rows*columns) + " nodes")
    println("Total of " + ivecs.ivecs.length + " input vectors")
    println("Using " + theta.getClass.getName)
    println("Using lattice " + l.getClass.getName)
    println("Jitter of " + jitter.toString)
    println("Base vectors are " + (if (corner) "at the corners" else "inside the corners"))
    (0 until baseVectors).foreach((i) => println("(" +
        ivecs.ivecs(i).map((x) => "%.2f".format(x)).reduce(_ + "," + _) 
        + ") has " + randomCounts(i) + " neighbors"))
    l.train(ivecs, epochs, its, theta.eval, progress.show)
    
    if (displayLattice) l.show(false)
    
    mm = l.neighborDist2()
    println("Minimum distance = " + mm._1 + ", maximum = " + mm._2)
    
    progress.show(l, 1000000, 1000000)
    
    umatrix = new Mat(2*(columns+1)*radius, 2*(rows+1)*radius, CvType.CV_8UC3, new Scalar(0,0,0))
    render = new UMatrixRender(radius, umatrix, mm._2)
    l.umatrixNodes(render.render)
    org.opencv.imgcodecs.Imgcodecs.imwrite(basePath + "umatrix-only.jpg", umatrix)
    l.members.flatten.foreach((n) => 
      org.opencv.imgproc.Imgproc.circle(umatrix, 
                                        new Point((2*n.x+1)*radius, (2*n.y+1)*radius), 
                                        (1.5*radius/2.0).toInt, 
                                        new Scalar(255.0*n.w(0), 255.0*n.w(1), 255.0*n.w(2)), 
                                        -1, 8, 0))

    org.opencv.imgcodecs.Imgcodecs.imwrite(basePath + "umatrix.jpg", umatrix)

    // TODO: testing
    fp = new File(basePath + "config.json")
    os = new FileOutputStream(fp)
    jw = Json.createWriter(os)
    jw.writeObject(serialize())
    jw.close()
  }
  /**
   * Create training vectors
   * 
   * Populate corner vectors, and then add random vectors clustered around them
   */
  def initTrainingVectors(rndCount: Int, jitter: Double): InputVectors = {
    val offset = if (corner) 0.0 else jitter
    var baseVecs = List(Array(1.0-offset, offset, offset, 0.5))
    if (baseVectors >= 2) baseVecs = baseVecs :+ Array(    offset, 1.0-offset,     offset, 0.5)
    if (baseVectors >= 3) baseVecs = baseVecs :+ Array(    offset,     offset, 1.0-offset, 0.5)
    if (baseVectors >= 4) baseVecs = baseVecs :+ Array(1.0-offset,     offset, 1.0-offset, 0.5)
    if (baseVectors >= 5) baseVecs = baseVecs :+ Array(    offset, 1.0-offset, 1.0-offset, 0.5)
    if (baseVectors >= 6) baseVecs = baseVecs :+ Array(1.0-offset, 1.0-offset,     offset, 0.5)
    if (baseVectors >= 7) baseVecs = baseVecs :+ Array(1.0-offset, 1.0-offset, 1.0-offset, 0.5)
    if (baseVectors >= 8) baseVecs = baseVecs :+ Array(    offset,     offset,     offset, 0.5)

    var ivecs = new InputVectors()
    randomCounts = ivecs.fillClusters(4, baseVecs, jitter, rndCount)
    ivecs.randomizeOneDimension(3, 0.0, 1.0)
  }
    
  /**
   * Create map describing command line parameters
   */
  def cmdLineOptions() : Map[String,(String,String)] = {
    val opts = Map(
        "-config" -> ("string", "full path to a configuration file"),
        "-rows" -> ("integer", "rows in the lattice"),
        "-cols" -> ("integer", "columns in the lattice"),
        "-layout" -> ("string", "hex or rect"),
        "-dim" -> ("integer", "dimension of the weight vectors"),
        "-random" -> ("integer", "number of random vectors to generate"),
        "-radius" -> ("integer", "size (in pixels) of circles to draw"),
        "-umatrix" -> ("null", "generate UMatrix for final networks"),
        "-epochs" -> ("integer", "total number of training epochs"),
        "-steps" -> ("integer", "number of training steps per epoch"),
        "-jitter" -> ("double", "amount to jitter random input vectors"),
        "-base" -> ("integer", "number of base vectors to generate"),
        "-displearn" -> ("null", "create samples of learning function"),
        "-disptime" -> ("null", "create samples of the time function"),
        "-start" -> ("double", "start value when generating learning function"),
        "-end" -> ("double", "end value when generating learning function"),
        "-step" -> ("double", "increment when generating learning function"),
        "-learn" -> ("string", "gaussian or bubble"),
        "-time" -> ("string", "linear, iot, or power"),
        "-show" -> ("null", "dump out network at completion of training"),
        "-path" -> ("string", "base path of the image file names"),
        "-template" -> ("string", "template of the image file name"),
        "-corner" -> ("null", "if present, put the base vectors at the corners of the cube"),
        "-markers" -> ("null", "show markers on the images where input vectors are located"),
        "-help" -> ("null", "show help message")
        )
    opts
  }
    
  /**
   * Parse command line
   */
  def parseCmdLine(args: Array[String], opts: Map[String,(String,String)]) : Unit = {
    var i: Int = 0
    var makeHex: Boolean = false
    var configPath: String = null
    
    while (i < args.length) {
      val xxx = opts.get(args(i)).get._1
      if (opts.contains(args(i)) && (i < args.length-1 || opts.get(args(i)).get._1 == "null")) {
        args(i) match {
          case "-config" => configPath = args(i+1)
          case "-rows" => rows = args(i+1).toInt
          case "-cols" => columns = args(i+1).toInt
          case "-layout" => makeHex = (args(i+1) == "hex")
          case "-dim" => dimension = args(i+1).toInt
          case "-random" => randomVecs = args(i+1).toInt
          case "-radius" => radius = args(i+1).toInt
          case "-umatrix" => createUmatrix = true 
          case "-epochs" => epochs = args(i+1).toInt
          case "-steps" => its = args(i+1).toInt
          case "-jitter" => jitter = args(i+1).toDouble
          case "-base" => baseVectors = args(i+1).toInt
          case "-displearn" => displayLearning = true
          case "-disptime" => displayTime = true
          case "-start" => start = args(i+1).toDouble
          case "-end" => end = args(i+1).toDouble
          case "-step" => step = args(i+1).toDouble
          case "-learn" => learn = args(i+1)
          case "-time" => time = args(i+1)
          case "-show" => displayLattice = true
          case "-path" => basePath = args(i+1)
          case "-corner" => corner = true
          case "-template" => fileTemplate = args(i+1)
          case "-markers" => showMarkers = true
        }
        i = i + (if (opts.get(args(i)).get._1 == "null") 1 else 2)
      } else {
        cmdLineShowHelp(opts)
        System.exit(1)
      }
    }
    
    if (configPath != null) {
      val fp: FileReader = new FileReader(new File(configPath))
      config = Json.createReader(fp).readObject()
      unserialize(config)
    } else {
      l = SomLatticeFactory.createLattice(if (makeHex) "hex" else "rect", rows, columns, dimension)      		
    }
    
    theta = learn match {
      case "gaussian" => time match {
        case "linear" => new GaussianLinear(rows, columns, epochs, its)
        case "iot" => new GaussianIoT(rows, columns, epochs, its)
        case "power" => new GaussianPower(rows, columns, epochs, its)
        case "_" => null
      }
      case "bubble" => time match {
        case "linear" => new BubbleLinear(rows, columns, epochs, its)
        case "iot" => new BubbleIoT(rows, columns, epochs, its)
        case "power" => new BubblePower(rows, columns, epochs, its)
        case "_" => null
      }
    }
    
    if (theta == null) {
      cmdLineShowHelp(opts)
      System.exit(1)
    }
    
    if (! basePath.endsWith("/")) basePath = basePath + "/"
  }
    
  /**
   * Show help for parameters
   */
  def cmdLineShowHelp(opts: Map[String,(String,String)]) {
    def fn(k: String) : Unit = {
      val v = opts.get(k)
      v match {
        case None => println("")
        case Some(v) => println("  " + k + " (" + v._1 + "): " + v._2)
      }
    }
    println("Usage: test")
    opts.keys.foreach((k: String) => fn(k))
  }
  
  /**
   * Serialize the current state
   */
  def serialize(fact: JsonBuilderFactory = null) : JsonObject = {
    var factory = if (fact == null) Json.createBuilderFactory(null) else fact
    // base vector counts
    val baseCountsB = factory.createArrayBuilder()
    randomCounts.foreach((x) => baseCountsB.add(x))

    var cfg = factory.createObjectBuilder()
      .add("rows", rows)
      .add("columns", columns)
      .add("dimension", dimension)
      .add("epochs", epochs)
      .add("its", its)
      .add("jitter", jitter)
      .add("corner", corner)
      .add("base-vectors", baseVectors)
      .add("random-counts", baseCountsB.build())
      // TODO: umatrix
      .add("training-set", ivecs.serialize(factory)/* TODO: ivecB.build() */)
      .add("base-path", basePath)
      .add("file-template", fileTemplate)
      .add("show-markers", showMarkers)
      .add("time", time)
      .add("learn", learn)
      .build()
    factory.createObjectBuilder()
      .add("config", cfg)
      .add("lattice", if (l==null) null else l.serialize(factory))
      .build()
  }
 
  /**
   * Unserialize the instance
   */
  def unserialize(masterCfg: JsonObject) = {
    val cfg: JsonObject = masterCfg.getJsonObject("config")
    try { rows = cfg.getInt("rows") } catch { case e: Exception => {} }
    try { columns = cfg.getInt("columns") } catch { case e: Exception => {} }
    try { dimension = cfg.getInt("dimension") } catch { case e: Exception => {} }
    try { epochs = cfg.getInt("epochs") } catch { case e: Exception => {} }
    try { its = cfg.getInt("its") } catch { case e: Exception => {} }
    if (cfg.getJsonNumber("jitter") != null) jitter = cfg.getJsonNumber("jitter").doubleValue()
    try { corner = cfg.getBoolean("corner") } catch { case e: Exception => {} }
    if (cfg.getString("base-path",null) != null) basePath = cfg.getString("base-path") 
    if (cfg.getString("file-template",null) != null) fileTemplate = cfg.getString("file-template")
    try { showMarkers = cfg.getBoolean("show-markers") } catch { case e: Exception => {} }
    if (cfg.getString("time",null) != null) time = cfg.getString("time")
    if (cfg.getString("learn",null) != null) learn = cfg.getString("learn")
    if (cfg.get("random-counts") != null) randomCounts = cfg.getJsonArray("random-counts").toArray().map(_.asInstanceOf[JsonNumber].intValue())
    if (cfg.get("training-set") != null) ivecs.unserialize(cfg)
/*        ivecs = cfg.getJsonArray("training-set").toArray()
                   .map((a:Object) => a.asInstanceOf[JsonArray].toArray()
                   .map(_.asInstanceOf[JsonNumber].doubleValue())).toList
*/
    val latticeCfg: JsonObject = masterCfg.getJsonObject("lattice")
    if (latticeCfg != null) l = SomLatticeFactory.createLattice(latticeCfg)
  }
  
  /**
   * Dump out the learning function to the stdout
   */
  def showLearningFunction() : Unit = {
    val totSteps = (end - start) / step;
    // TODO: make time steps and epochs configurable
    for (t <- 1 until 202 by 100) {
      println(theta.toString + ", epoch = 1, time step = " + t)
      val f = eval_function(start, end, step, 1, t, theta.eval)
      f.foreach((a) => println(a.map((x) => x._3.toString).reduceLeft(_ + "," + _)))      
    }
  }
  
  /**
   * Dump out the time function to the stdout
   */
  def showTimeFunction() : Unit = {
    val r:Array[Int] = (start.toInt to end.toInt by 1).toArray
    val y:Array[(Int,Double)] = time match {
      case "linear" => r.map((x) => (x, theta.lrLinear(x)))
      case "iot" => r.map((x) => (x, theta.lrIoT(x)))
      case "power" => r.map((x) => (x, theta.lrPowerSeries(x)))
    }
    println(theta.toString + "time function")
    y.foreach((a) => println(a._1 + "," + a._2))
  }

  def eval_function(start: Double, end: Double, step: Double, epoch: Int, ts: Int, 
    fn: (Double, Double, Double, Double, Int, Int) => Double): Array[Array[(Double,Double,Double)]] = {
    val r = (start to end by step).toArray
    r.map(p => r.map(o => (p,o,fn(p,o,0.0,0.0,epoch,ts))))
  }
}

class UMatrixRender(radius: Int, u: Mat, mxd2: Double) {
  def render(x: Double, y: Double, d2: Double) : Unit = {
    val pt = new Point(radius+2*radius*x, radius+2*radius*y)
    val grey = 255.0 * d2 / mxd2
    org.opencv.imgproc.Imgproc.circle(u, pt, radius/2, new Scalar(grey, grey, grey), -1, 8, 0)
  }
  def showDistance(x: Double, y: Double, d2: Double) : Unit = {
    println("Distance at (" + x + "," + y + ") is " + d2)
  }
}

class Progress(epochStep: Int, iterationStep: Int, radius: Int, img: Mat, out_template: String, iv: InputVectors) {
  val textMarg = 20
  def show(l: SomLattice, epoch: Int, step: Int): Unit =
  {
    if (epoch == -1 || (epochStep == 0) || ((epoch % epochStep) == 0)) {
      if (step == -1 || 
          (iterationStep == 0 && step == 0) || 
          (iterationStep == 1) || 
          ((iterationStep > 1) && ((step % iterationStep) == 0))) {
        org.opencv.imgproc.Imgproc.rectangle(img, new Point(0.0,0.0), new Point(img.width()-1,img.height()-1), new Scalar(0.0, 0.0, 0.0), -1)
        l.members.flatten.foreach((n: SomNode) => org.opencv.imgproc.Imgproc.circle(img, new Point((2*(n.x)+1)*radius, (2*n.y+1)*radius), radius, color_node(n.w), -1, 8, 0))
        val white = new Scalar(255.0,255.0,255.0)
        if (iv != null)
          iv.ivecs.foreach((v) => {
            val xy = l.selectRepPt(v)
            val d = radius/2
            val black = new Scalar(0.0, 0.0, 0.0)
            val p1 = new Point(2*(xy._1+1)*radius-d, 2*(xy._2+1)*radius-d)
            val p2 = new Point(2*(xy._1+1)*radius+d, 2*(xy._2+1)*radius+d)
            val p3 = new Point(2*(xy._1+1)*radius+d, 2*(xy._2+1)*radius-d)
            val p4 = new Point(2*(xy._1+1)*radius-d, 2*(xy._2+1)*radius+d)
            org.opencv.imgproc.Imgproc.line(img, p1, p2, black, 8)
            org.opencv.imgproc.Imgproc.line(img, p3, p4, black, 8)
            org.opencv.imgproc.Imgproc.line(img, p1, p2, white, 2)
            org.opencv.imgproc.Imgproc.line(img, p3, p4, white, 2)
          })
        Imgproc.putText(img, "EPOCH: " + epoch + "  STEP: " + step, new Point(textMarg, img.height()-textMarg), Core.FONT_HERSHEY_SIMPLEX, 2, white, 3)
        val path = out_template.replaceAll("%E", epoch.toString).replaceAll("%T", step.toString)
        org.opencv.imgcodecs.Imgcodecs.imwrite(path, img)
      }
    }
  }

  def color_node(w: Array[Double]): Scalar = {
    new Scalar(255.0*w(0), 255.0*w(1), 255.0*w(2))
  }
}
      
class GaussianLinear(rows: Int, columns: Int, maxEpochs:Int, maxSteps: Int) 
  extends TrainingFunction(rows,columns,maxEpochs,maxSteps) {
  override def eval(x: Double, y: Double, cx: Double, cy: Double, epoch: Int, ts: Int): Double = {
    fGaussian(x, y, cx, cy, epoch, ts) * lrLinear(ts)
  }
}

class GaussianIoT(rows: Int, columns: Int, maxEpochs: Int, maxSteps: Int)
  extends TrainingFunction(rows,columns,maxEpochs,maxSteps) {
  override def eval(x: Double, y: Double, cx: Double, cy: Double, epoch: Int, ts: Int): Double = {
    fGaussian(x, y, cx, cy, epoch, ts) * lrIoT(ts)
  }
}

class GaussianPower(rows: Int, columns: Int, maxEpochs: Int, maxSteps: Int)
  extends TrainingFunction(rows,columns,maxEpochs,maxSteps) {
  override def eval(x: Double, y: Double, cx: Double, cy: Double, epoch: Int, ts: Int): Double = {
    fGaussian(x, y, cx, cy, epoch, ts) * lrPowerSeries(ts)
  }
}

class BubbleLinear(rows: Int, columns: Int, maxEpochs:Int, maxSteps: Int) 
  extends TrainingFunction(rows,columns,maxEpochs,maxSteps) {
  override def eval(x: Double, y: Double, cx: Double, cy: Double, epoch: Int, ts: Int): Double = {
    fBubble(x, y, cx, cy, epoch, ts) * lrLinear(ts)
  }
}

class BubbleIoT(rows: Int, columns: Int, maxEpochs: Int, maxSteps: Int)
  extends TrainingFunction(rows,columns,maxEpochs,maxSteps) {
  override def eval(x: Double, y: Double, cx: Double, cy: Double, epoch: Int, ts: Int): Double = {
    fBubble(x, y, cx, cy, epoch, ts) * lrIoT(ts)
  }
}

class BubblePower(rows: Int, columns: Int, maxEpochs: Int, maxSteps: Int)
  extends TrainingFunction(rows,columns,maxEpochs,maxSteps) {
  override def eval(x: Double, y: Double, cx: Double, cy: Double, epoch: Int, ts: Int): Double = {
    fBubble(x, y, cx, cy, epoch, ts) * lrPowerSeries(ts)
  }
}