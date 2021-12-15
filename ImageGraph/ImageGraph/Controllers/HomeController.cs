using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading.Tasks;
using ImageGraph.Models;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;

namespace ImageGraph.Controllers
{
   public class HomeController : Controller
   {

      private const string ContentType = "application/json";

      private readonly ILogger<HomeController> _logger;

      public HomeController(ILogger<HomeController> logger)
      {
         _logger = logger;
      }

      public IActionResult Index()
      {
         return View();
      }

      [HttpPost("path")]
      public async Task<IActionResult> ShortestPath([FromBody] object data)
      {
         var dataString = data.ToString();
         var model = JsonConvert.DeserializeObject<ShortestPathModel>(dataString);

         var body = new
         {
            fileName = model.FileName,
            maxX = model.MaxX,
            maxY = model.MaxY,
            cors = model.Cors
         };

         var pathRequest = new HttpRequestMessage(HttpMethod.Post, "http://127.0.0.1:5000/shortestPath");
         var serializedBody = JsonConvert.SerializeObject(body);
         var content = new StringContent(serializedBody);

         content.Headers.ContentType = MediaTypeHeaderValue.Parse(ContentType);
         pathRequest.Content = content;

         var client = new HttpClient();
         var result = await client.SendAsync(pathRequest);

         var resultString = await result.Content.ReadAsStringAsync();

         var resmodel = JsonConvert.DeserializeObject<ShortestPathResponseModel>(resultString);
         resmodel.Len *= model.Scale ?? 3.6329;

         var responseModelMapped = new ShortestPathResponseModelApi
         {
            Img = resmodel.Img,
            Len = (Math.Truncate(resmodel.Len * 100) / 100).ToString()
         };

         return Ok(responseModelMapped);
      }

      [HttpPost("file")]
      public async Task<IActionResult> File(IFormFile file)
      {
         if (file.Length > 0)
         {
            using (Stream fileStream = new FileStream($"..\\..\\Roads-Segmentation\\Predictions\\input\\{file.FileName}", FileMode.Create))
            {
               await file.CopyToAsync(fileStream);
            }
         }

         return Ok();
      }
   }
}
