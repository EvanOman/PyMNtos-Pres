//	Original source:
//	http://nuclear.mutantstargoat.com/articles/sdr_fract/


uniform sampler1D tex;
uniform vec2 c;
uniform int iter;

void main() {
	vec2 z;
	z.x = 3.0 * (gl_TexCoord[0].x - 0.5);
	z.y = 2.0 * (gl_TexCoord[0].y - 0.5);

	int i;
	for(i=0; i<iter; i++) {
		// These lines are my main contribution, they implement the map 
		// x^2 + c + .001/conj(x)^2

		float x = c.x + z.x * z.x - z.y * z.y + (.001 * z.x * z.x)/((z.x * z.x + z.y *z.y) * (z.x * z.x + z.y *z.y)) - (.001 * z.y * z.y)/((z.x * z.x + z.y *z.y)*(z.x * z.x + z.y *z.y));
		float y = 2*z.x*z.y + (.002* z.x * z.y)/((z.x * z.x + z.y *z.y) * (z.x * z.x + z.y *z.y)) + c.y;

		if((x * x + y * y) > 4.0) break;
		z.x = x;
		z.y = y;
	}

	float scale = 1.0 -  float(i)/float(iter + 1);
//	gl_FragColor = vec4(scale, 0.0, scale, 1.0);

	gl_FragColor = texture1D(tex, i == 0 ? .99999 : scale);
}
