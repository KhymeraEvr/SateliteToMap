namespace ImageGraph.Models
{
   public class ShortestPathModel
   {
      public List<Point> Cors { get; set; }

      public string MaxX { get; set; }

      public string MaxY { get; set; }

      public string FileName { get; set; }
   }

   public class Point
   {
      public string X { get; set; }

      public string Y { get; set; }
   }
}
