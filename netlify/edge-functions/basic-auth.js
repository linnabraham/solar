export default async (request, context) => {
  const expectedUser = "admin";
  const expectedPassword = Netlify.env.get("SITE_PASSWORD"); 
  
  const authHeader = request.headers.get("authorization");
  const expectedAuth = `Basic ${btoa(`${expectedUser}:${expectedPassword}`)}`;

  if (authHeader !== expectedAuth) {
    return new Response("Unauthorized", {
      status: 401,
      headers: { "WWW-Authenticate": 'Basic realm="Secure Area"' },
    });
  }
  return context.next();
};

export const config = { path: "/*" };
